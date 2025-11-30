from dataclasses import asdict

import gstaichi as ti

from slobot.rigid_body.configuration import Configuration, rigid_body_configuration
from slobot.rigid_body.state import ConfigurationState, create_entity_state, from_dict


def _infer_nested_list_shape(data):
    """Return the tuple shape of a nested Python list."""
    if isinstance(data, list):
        length = len(data)
        if length == 0:
            return (0,)
        first_shape = _infer_nested_list_shape(data[0])
        for item in data[1:]:
            if _infer_nested_list_shape(item) != first_shape:
                raise ValueError("Non-uniform nested list structure for Taichi ndarray conversion")
        return (length,) + first_shape
    return ()


def _populate_ndarray_from_list(arr, data, index_prefix=()):
    """Populate a Taichi ndarray with values from a nested list."""
    if isinstance(data, list):
        for idx, item in enumerate(data):
            _populate_ndarray_from_list(arr, item, index_prefix + (idx,))
    else:
        arr[index_prefix] = data


def make_taichi_vector_factory():
    """Create a vector factory that converts lists to Taichi ndarrays."""
    def _factory(data: list):
        shape = _infer_nested_list_shape(data)
        arr_ti = ti.ndarray(dtype=ti.f64, shape=shape)
        _populate_ndarray_from_list(arr_ti, data)
        return arr_ti
    return _factory

@ti.data_oriented
class TaichiSolver:
    # Numerical epsilon threshold for detecting near-zero quaternions/vectors
    EPS = 1e-8
    QUAT0 = [1.0, 0, 0, 0]
    
    def __init__(self, arch=ti.cpu) -> None:
        """Initialize Taichi solver.
        
        Args:
            arch: Taichi architecture (ti.cpu, ti.gpu, ti.cuda, etc.)
        """
        # Initialize Taichi
        ti.init(arch=arch, default_fp=ti.f64)
        self.arch = arch
        self.config: Configuration = rigid_body_configuration
        
        # Initialize entity states using factory function
        self.previous_entity = create_entity_state()
        self.current_entity = create_entity_state()
        
        # Create ConfigurationState with Taichi ndarrays using from_dict
        config_dict = asdict(self.config.config_state)
        taichi_factory = make_taichi_vector_factory()
        self.config_state: ConfigurationState = from_dict(ConfigurationState, config_dict, taichi_factory)
        
        # Drop base link from config_state fields (excluding first element/row)
        self.drop_base_link()
        self.loss = ti.field(dtype=ti.f64, shape=())
        self.trainable_config_fields = []

    def drop_base_link(self):
        """Create versions without base link in config_state (excluding first element/row)."""
        # Create versions without base link (excluding first element/row)
        # For Taichi ndarrays, we need to create new arrays with sliced data
        self.config_state.link_initial_quat_no_base = self._slice_ndarray(self.config_state.link_initial_quat, 1, None)
        self.config_state.link_initial_pos_no_base = self._slice_ndarray(self.config_state.link_initial_pos, 1, None)
        self.config_state.link_mass_no_base = self._slice_ndarray(self.config_state.link_mass, 1, None)
        self.config_state.link_inertia_no_base = self._slice_ndarray(self.config_state.link_inertia, 1, None)
        self.config_state.link_inertial_quat_no_base = self._slice_ndarray(self.config_state.link_inertial_quat, 1, None)
        self.config_state.link_inertial_pos_no_base = self._slice_ndarray(self.config_state.link_inertial_pos, 1, None)

    def enable_gradients_for(self, field_names):
        """Enable Taichi autodiff for selected configuration fields."""
        self.trainable_config_fields = list(field_names)
        for name in field_names:
            self._enable_gradient_for_field(name)

    def _enable_gradient_for_field(self, field_name: str):
        field = getattr(self.config_state, field_name)
        grad_enabled = ti.ndarray(dtype=field.dtype, shape=field.shape, needs_grad=True)
        grad_enabled.from_numpy(field.to_numpy())
        setattr(self.config_state, field_name, grad_enabled)
    
    @ti.kernel
    def _slice_ndarray_kernel_1d(self, arr: ti.types.ndarray(), out: ti.types.ndarray(), start: int):
        """Taichi kernel for slicing 1D array."""
        for i in range(out.shape[0]):
            out[i] = arr[start + i]
    
    @ti.kernel
    def _slice_ndarray_kernel_2d(self, arr: ti.types.ndarray(), out: ti.types.ndarray(), start: int):
        """Taichi kernel for slicing 2D array."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = arr[start + i, j]
    
    @ti.kernel
    def _slice_ndarray_kernel_3d(self, arr: ti.types.ndarray(), out: ti.types.ndarray(), start: int):
        """Taichi kernel for slicing 3D array."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                for k in range(out.shape[2]):
                    out[i, j, k] = arr[start + i, j, k]
    
    def _slice_ndarray(self, arr: ti.ndarray, start: int, stop: int):
        """Slice a Taichi ndarray."""
        if stop is None:
            stop = arr.shape[0]
        slice_size = stop - start
        
        if len(arr.shape) == 1:
            out = ti.ndarray(dtype=ti.f64, shape=(slice_size,))
            self._slice_ndarray_kernel_1d(arr, out, start)
        elif len(arr.shape) == 2:
            out = ti.ndarray(dtype=ti.f64, shape=(slice_size, arr.shape[1]))
            self._slice_ndarray_kernel_2d(arr, out, start)
        elif len(arr.shape) == 3:
            out = ti.ndarray(dtype=ti.f64, shape=(slice_size, arr.shape[1], arr.shape[2]))
            self._slice_ndarray_kernel_3d(arr, out, start)
        else:
            raise ValueError(f"Unsupported array dimension: {len(arr.shape)}")
        return out

    @ti.kernel
    def _reset_loss_kernel(self):
        """Reset accumulated loss."""
        self.loss[None] = 0.0

    @ti.kernel
    def _accumulate_loss_kernel(self, sim_pos: ti.types.ndarray(), target_pos: ti.types.ndarray(), frame_weight: ti.f64):
        """Accumulate frame loss using L2 norm."""
        acc = 0.0
        for i in range(sim_pos.shape[0]):
            diff = sim_pos[i] - target_pos[i]
            acc += diff * diff
        self.loss[None] += ti.sqrt(acc) * frame_weight

    @ti.kernel
    def _zero_ndarray_kernel(self, arr: ti.types.ndarray()):
        """Zero out a Taichi ndarray."""
        for I in ti.grouped(arr):
            arr[I] = 0.0

    def reset_loss(self):
        """Reset the scalar loss accumulator."""
        self._reset_loss_kernel()

    def accumulate_position_loss(self, target_pos: ti.ndarray, frame_weight: float):
        """Accumulate L2 position error against the provided target."""
        self._accumulate_loss_kernel(self.current_entity.joint.pos, target_pos, frame_weight)

    def zero_param_grads(self):
        """Zero gradients for all enabled configuration parameters."""
        for name in self.trainable_config_fields:
            grad = getattr(self.config_state, name).grad
            self._zero_ndarray_kernel(grad)
    
    @ti.kernel
    def _list_to_ndarray_kernel_1d(self, data: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel to copy 1D list data to ndarray."""
        for i in range(out.shape[0]):
            out[i] = data[i]
    
    @ti.kernel
    def _list_to_ndarray_kernel_2d(self, data: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel to copy 2D list data to ndarray."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = data[i, j]
    
    @ti.kernel
    def _list_to_ndarray_kernel_3d(self, data: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel to copy 3D list data to ndarray."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                for k in range(out.shape[2]):
                    out[i, j, k] = data[i, j, k]
    
    @ti.kernel
    def _copy_from_list_1d_kernel(self, data: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel to copy 1D list data to ndarray."""
        for i in range(out.shape[0]):
            out[i] = data[i]
    
    @ti.kernel
    def _copy_from_list_2d_kernel(self, data: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel to copy 2D list data to ndarray."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = data[i, j]
    
    @ti.kernel
    def _copy_from_list_3d_kernel(self, data: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel to copy 3D list data to ndarray."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                for k in range(out.shape[2]):
                    out[i, j, k] = data[i, j, k]
    
    @ti.kernel
    def _subtract_kernel_1d(self, a: ti.types.ndarray(), b: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for subtracting 1D arrays."""
        for i in range(out.shape[0]):
            out[i] = a[i] - b[i]
    
    @ti.kernel
    def _subtract_kernel_2d(self, a: ti.types.ndarray(), b: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for subtracting 2D arrays."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = a[i, j] - b[i, j]
    
    @ti.kernel
    def _subtract_kernel_3d(self, a: ti.types.ndarray(), b: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for subtracting 3D arrays."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                for k in range(out.shape[2]):
                    out[i, j, k] = a[i, j, k] - b[i, j, k]
    
    @ti.kernel
    def _subtract_broadcast_kernel(self, a: ti.types.ndarray(), b: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for subtracting 1D array from 2D array (broadcasting)."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = a[i, j] - b[j]
    
    def _taichi_subtract(self, a, b):
        """Subtract two Taichi arrays."""
        result = ti.ndarray(dtype=ti.f64, shape=a.shape)
        
        if len(a.shape) == 1 and len(b.shape) == 1:
            self._subtract_kernel_1d(a, b, result)
        elif len(a.shape) == 2 and len(b.shape) == 2:
            self._subtract_kernel_2d(a, b, result)
        elif len(a.shape) == 3 and len(b.shape) == 3:
            self._subtract_kernel_3d(a, b, result)
        elif len(a.shape) == 2 and len(b.shape) == 1:
            # Broadcast: subtract 1D from each row of 2D
            self._subtract_broadcast_kernel(a, b, result)
        else:
            raise ValueError(f"Unsupported shapes for subtraction: {a.shape} and {b.shape}")
        return result
    
    @ti.kernel
    def _add_kernel_1d(self, a: ti.types.ndarray(), b: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for adding 1D arrays."""
        for i in range(out.shape[0]):
            out[i] = a[i] + b[i]
    
    @ti.kernel
    def _add_kernel_2d(self, a: ti.types.ndarray(), b: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for adding 2D arrays."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = a[i, j] + b[i, j]
    
    @ti.kernel
    def _add_kernel_3d(self, a: ti.types.ndarray(), b: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for adding 3D arrays."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                for k in range(out.shape[2]):
                    out[i, j, k] = a[i, j, k] + b[i, j, k]
    
    @ti.kernel
    def _add_broadcast_1d_to_2d_kernel(self, a: ti.types.ndarray(), b: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for adding 1D array to each row of 2D array (broadcasting)."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = a[i, j] + b[j]
    
    def _taichi_add(self, a, b):
        """Add two Taichi arrays."""
        # Handle broadcasting
        if len(a.shape) == 2 and len(b.shape) == 1:
            result = ti.ndarray(dtype=ti.f64, shape=a.shape)
            self._add_broadcast_1d_to_2d_kernel(a, b, result)
            return result
        elif len(a.shape) == 1 and len(b.shape) == 2:
            result = ti.ndarray(dtype=ti.f64, shape=b.shape)
            self._add_broadcast_1d_to_2d_kernel(b, a, result)
            return result
        
        result = ti.ndarray(dtype=ti.f64, shape=a.shape)
        if len(a.shape) == 1:
            self._add_kernel_1d(a, b, result)
        elif len(a.shape) == 2:
            self._add_kernel_2d(a, b, result)
        elif len(a.shape) == 3:
            self._add_kernel_3d(a, b, result)
        else:
            raise ValueError(f"Unsupported shape for addition: {a.shape}")
        return result
    
    @ti.kernel
    def _multiply_kernel_1d(self, a: ti.types.ndarray(), b: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for multiplying 1D arrays element-wise."""
        for i in range(out.shape[0]):
            out[i] = a[i] * b[i]
    
    @ti.kernel
    def _multiply_kernel_2d(self, a: ti.types.ndarray(), b: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for multiplying 2D arrays element-wise."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = a[i, j] * b[i, j]
    
    def _taichi_multiply(self, a, b):
        """Multiply two Taichi arrays element-wise."""
        result = ti.ndarray(dtype=ti.f64, shape=a.shape)
        if len(a.shape) == 1:
            self._multiply_kernel_1d(a, b, result)
        else:
            self._multiply_kernel_2d(a, b, result)
        return result

    # ----------------------------- basic helpers -----------------------------
    @ti.kernel
    def _max_abs_error_kernel_1d(self, actual: ti.types.ndarray(), expected: ti.types.ndarray()) -> ti.f64:
        """Taichi kernel for max absolute error (1D)."""
        max_err = 0.0
        for i in range(actual.shape[0]):
            err = ti.abs(actual[i] - expected[i])
            if err > max_err:
                max_err = err
        return max_err
    
    @ti.kernel
    def _max_abs_error_kernel_2d(self, actual: ti.types.ndarray(), expected: ti.types.ndarray()) -> ti.f64:
        """Taichi kernel for max absolute error (2D)."""
        max_err = 0.0
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                err = ti.abs(actual[i, j] - expected[i, j])
                if err > max_err:
                    max_err = err
        return max_err
    
    def max_abs_error(self, actual, expected):
        """Compute maximum absolute error between two arrays."""
        if len(actual.shape) == 1:
            return self._max_abs_error_kernel_1d(actual, expected)
        else:
            return self._max_abs_error_kernel_2d(actual, expected)

    @ti.kernel
    def _cross_product_kernel(self, vecs1: ti.types.ndarray(), vecs2: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for cross product."""
        for i in range(vecs1.shape[0]):
            out[i, 0] = vecs1[i, 1] * vecs2[i, 2] - vecs1[i, 2] * vecs2[i, 1]
            out[i, 1] = vecs1[i, 2] * vecs2[i, 0] - vecs1[i, 0] * vecs2[i, 2]
            out[i, 2] = vecs1[i, 0] * vecs2[i, 1] - vecs1[i, 1] * vecs2[i, 0]

    def cross_product(self, vecs1, vecs2):
        """Compute cross product of two vector arrays."""
        out = ti.ndarray(dtype=ti.f64, shape=vecs1.shape)
        self._cross_product_kernel(vecs1, vecs2, out)
        return out

    @ti.kernel
    def _scalar_product_kernel(self, vecs1: ti.types.ndarray(), vecs2: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for scalar product (dot product per row)."""
        for i in range(vecs1.shape[0]):
            out[i] = vecs1[i, 0] * vecs2[i, 0] + vecs1[i, 1] * vecs2[i, 1] + vecs1[i, 2] * vecs2[i, 2]

    def scalar_product(self, vecs1, vecs2):
        """Compute scalar product (dot product) of two vector arrays."""
        out = ti.ndarray(dtype=ti.f64, shape=(vecs1.shape[0],))
        self._scalar_product_kernel(vecs1, vecs2, out)
        return out

    @ti.kernel
    def _outer_product_kernel(self, vec1: ti.types.ndarray(), vec2: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for outer product."""
        for i in range(vec1.shape[0]):
            for j in range(vec2.shape[0]):
                out[i, j] = vec1[i] * vec2[j]
    
    def outer_product(self, vec1, vec2):
        """Compute outer product of two vectors."""
        out = ti.ndarray(dtype=ti.f64, shape=(vec1.shape[0], vec2.shape[0]))
        self._outer_product_kernel(vec1, vec2, out)
        return out

    @ti.kernel
    def _cumulative_sum_kernel(self, vecs: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for cumulative sum."""
        for i in range(vecs.shape[0]):
            if i == 0:
                for j in range(vecs.shape[1]):
                    out[i, j] = vecs[i, j]
            else:
                for j in range(vecs.shape[1]):
                    out[i, j] = out[i-1, j] + vecs[i, j]

    def cumulative_sum(self, vecs):
        """Compute cumulative sum along first axis."""
        out = ti.ndarray(dtype=ti.f64, shape=vecs.shape)
        self._cumulative_sum_kernel(vecs, out)
        return out

    @ti.kernel
    def _reverse_cumulative_sum_kernel_1d(self, vecs: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for reverse cumulative sum (1D arrays)."""
        n = vecs.shape[0]
        for i in range(n):
            idx = n - 1 - i
            if i == 0:
                out[idx] = vecs[idx]
            else:
                out[idx] = out[idx + 1] + vecs[idx]
    
    @ti.kernel
    def _reverse_cumulative_sum_kernel_2d(self, vecs: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for reverse cumulative sum (2D arrays)."""
        n = vecs.shape[0]
        for i in range(n):
            idx = n - 1 - i
            if i == 0:
                for j in range(vecs.shape[1]):
                    out[idx, j] = vecs[idx, j]
            else:
                for j in range(vecs.shape[1]):
                    out[idx, j] = out[idx + 1, j] + vecs[idx, j]
    
    @ti.kernel
    def _reverse_cumulative_sum_kernel_3d(self, vecs: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for reverse cumulative sum (3D arrays)."""
        n = vecs.shape[0]
        for i in range(n):
            idx = n - 1 - i
            if i == 0:
                for j in range(vecs.shape[1]):
                    for k in range(vecs.shape[2]):
                        out[idx, j, k] = vecs[idx, j, k]
            else:
                for j in range(vecs.shape[1]):
                    for k in range(vecs.shape[2]):
                        out[idx, j, k] = out[idx + 1, j, k] + vecs[idx, j, k]
    
    def reverse_cumulative_sum(self, vecs):
        """Compute reverse cumulative sum."""
        out = ti.ndarray(dtype=ti.f64, shape=vecs.shape)
        
        if len(vecs.shape) == 1:
            self._reverse_cumulative_sum_kernel_1d(vecs, out)
        elif len(vecs.shape) == 2:
            self._reverse_cumulative_sum_kernel_2d(vecs, out)
        elif len(vecs.shape) == 3:
            self._reverse_cumulative_sum_kernel_3d(vecs, out)
        else:
            raise ValueError(f"Unsupported array dimension: {len(vecs.shape)}")
        
        return out

    @ti.kernel
    def _multiply_matrix_by_vector_kernel(self, m: ti.types.ndarray(), u: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for matrix-vector multiplication."""
        for i in range(m.shape[0]):
            for j in range(m.shape[2]):
                out[i, j] = 0.0
                for k in range(m.shape[1]):
                    out[i, j] += m[i, k, j] * u[i, k]

    def multiply_matrix_by_vector(self, m, u):
        """Multiply matrix by vector."""
        out = ti.ndarray(dtype=ti.f64, shape=(m.shape[0], m.shape[2]))
        self._multiply_matrix_by_vector_kernel(m, u, out)
        return out

    @ti.kernel
    def _multiply_scalar_by_vector_kernel(self, c: ti.types.ndarray(), u: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for scalar-vector multiplication."""
        for i in range(c.shape[0]):
            for j in range(u.shape[1]):
                out[i, j] = c[i] * u[i, j]

    def multiply_scalar_by_vector(self, c, u):
        """Multiply scalar array by vector array."""
        out = ti.ndarray(dtype=ti.f64, shape=u.shape)
        self._multiply_scalar_by_vector_kernel(c, u, out)
        return out

    @ti.kernel
    def _multiply_scalar_by_matrix_kernel(self, c: ti.types.ndarray(), m: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for scalar-matrix multiplication."""
        for i in range(c.shape[0]):
            for j in range(m.shape[1]):
                for k in range(m.shape[2]):
                    out[i, j, k] = c[i] * m[i, j, k]

    def multiply_scalar_by_matrix(self, c, m):
        """Multiply scalar array by matrix array."""
        out = ti.ndarray(dtype=ti.f64, shape=m.shape)
        self._multiply_scalar_by_matrix_kernel(c, m, out)
        return out

    @ti.kernel
    def _shift_bottom_kernel(self, A: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for shift bottom operation."""
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i == 0:
                    out[i, j] = 0.0
                else:
                    out[i, j] = A[i - 1, j]
    
    def shift_bottom(self, A):
        """Shift array down by one row, padding top with zeros."""
        out = ti.ndarray(dtype=ti.f64, shape=A.shape)
        self._shift_bottom_kernel(A, out)
        return out

    @ti.kernel
    def _hhT_batch_kernel(self, vecs: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for hhT batch computation."""
        for i in range(vecs.shape[0]):
            norm = vecs[i, 0] * vecs[i, 0] + vecs[i, 1] * vecs[i, 1] + vecs[i, 2] * vecs[i, 2]
            for j in range(3):
                for k in range(3):
                    if j == k:
                        out[i, j, k] = norm - vecs[i, j] * vecs[i, k]
                    else:
                        out[i, j, k] = -vecs[i, j] * vecs[i, k]

    def hhT_batch(self, vecs):
        """Compute hhT batch."""
        out = ti.ndarray(dtype=ti.f64, shape=(vecs.shape[0], 3, 3))
        self._hhT_batch_kernel(vecs, out)
        return out

    @ti.kernel
    def _matvec_kernel(self, m: ti.types.ndarray(), v: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for matrix-vector multiplication."""
        for i in range(m.shape[0]):
            out[i] = 0.0
            for j in range(m.shape[1]):
                out[i] += m[i, j] * v[j]
    
    def matvec(self, m, v):
        """Matrix-vector multiplication."""
        out = ti.ndarray(dtype=ti.f64, shape=(m.shape[0],))
        self._matvec_kernel(m, v, out)
        return out

    @ti.kernel
    def _init_augmented_matrix_kernel(self, m: ti.types.ndarray(), b: ti.types.ndarray(), aug: ti.types.ndarray()):
        """Taichi kernel to initialize augmented matrix."""
        n = m.shape[0]
        for i in range(n):
            for j in range(n):
                aug[i, j] = m[i, j]
            aug[i, n] = b[i]
    
    @ti.kernel
    def _gaussian_elimination_kernel(self, aug: ti.types.ndarray(), x: ti.types.ndarray()):
        """Taichi kernel for Gaussian elimination with partial pivoting."""
        n = aug.shape[0]
        singular = False
        
        # Forward elimination with partial pivoting
        for i in range(n):
            # Find pivot
            max_row = i
            max_val = ti.abs(aug[i, i])
            for k in range(i + 1, n):
                if ti.abs(aug[k, i]) > max_val:
                    max_val = ti.abs(aug[k, i])
                    max_row = k
            
            # Swap rows
            if max_row != i:
                for j in range(n + 1):
                    temp = aug[i, j]
                    aug[i, j] = aug[max_row, j]
                    aug[max_row, j] = temp
            
            # Check for singularity
            pivot = aug[i, i]
            if ti.abs(pivot) < 1e-10:
                # Singular matrix, set solution to zero
                singular = True
            
            # Eliminate (only if not singular so far)
            if not singular:
                for k in range(i + 1, n):
                    factor = aug[k, i] / pivot
                    for j in range(i, n + 1):
                        aug[k, j] -= factor * aug[i, j]
        
        # Back substitution (only if not singular)
        if not singular:
            for i_idx in range(n):
                i = n - 1 - i_idx  # Reverse iteration
                x[i] = aug[i, n]
                for j in range(i + 1, n):
                    x[i] -= aug[i, j] * x[j]
                x[i] /= aug[i, i]
        else:
            # Set solution to zero if singular
            for j in range(n):
                x[j] = 0.0
    
    def linalg_solve(self, m, b):
        """Solve linear system using Gaussian elimination."""
        n = m.shape[0]
        
        # Create augmented matrix outside kernel
        aug = ti.ndarray(dtype=ti.f64, shape=(n, n + 1))
        self._init_augmented_matrix_kernel(m, b, aug)
        
        result_ti = ti.ndarray(dtype=ti.f64, shape=b.shape)
        self._gaussian_elimination_kernel(aug, result_ti)
        return result_ti

    @ti.kernel
    def _clip_kernel_1d(self, x: ti.types.ndarray(), min_v: ti.types.ndarray(), max_v: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for clipping 1D values."""
        for i in range(x.shape[0]):
            out[i] = ti.max(min_v[i], ti.min(max_v[i], x[i]))
    
    @ti.kernel
    def _clip_kernel_2d(self, x: ti.types.ndarray(), min_v: ti.types.ndarray(), max_v: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for clipping 2D values."""
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[i, j] = ti.max(min_v[i], ti.min(max_v[i], x[i, j]))
    
    def clip(self, x, min_v, max_v):
        """Clip values to range."""
        out = ti.ndarray(dtype=ti.f64, shape=x.shape)
        
        if len(x.shape) == 1:
            self._clip_kernel_1d(x, min_v, max_v, out)
        else:
            self._clip_kernel_2d(x, min_v, max_v, out)
        return out

    @ti.kernel
    def _tile_row_kernel(self, row: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for tiling a row."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if row.shape[0] > j:
                    out[i, j] = row[j]
                else:
                    out[i, j] = row[0]  # Handle 1D case
    
    def tile_row(self, row):
        """Tile a row to match DOFs."""
        # Handle both 1D and 2D row arrays
        if len(row.shape) == 1:
            row_size = row.shape[0]
        else:
            row_size = row.shape[1] if row.shape[0] == 1 else row.shape[0]
        out = ti.ndarray(dtype=ti.f64, shape=(self.config.dofs, row_size))
        self._tile_row_kernel(row, out)
        return out

    # ----------------------------- quaternion ops ----------------------------
    @ti.kernel
    def _normalize_quat_kernel(self, quats: ti.types.ndarray(), norms: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for normalizing quaternions."""
        for i in range(quats.shape[0]):
            norm = ti.sqrt(quats[i, 0] * quats[i, 0] + quats[i, 1] * quats[i, 1] + 
                          quats[i, 2] * quats[i, 2] + quats[i, 3] * quats[i, 3])
            norms[i] = norm
            if norm < self.EPS:
                # Use identity quaternion
                out[i, 0] = 1.0
                out[i, 1] = 0.0
                out[i, 2] = 0.0
                out[i, 3] = 0.0
            else:
                for j in range(4):
                    out[i, j] = quats[i, j] / norm
    
    @ti.kernel
    def _transform_by_quat_kernel(self, v: ti.types.ndarray(), quat: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for quaternion transformation (from PyTorch implementation)."""
        for i in range(v.shape[0]):
            v_x = v[i, 0]
            v_y = v[i, 1]
            v_z = v[i, 2]
            
            q_w = quat[i, 0]
            q_x = quat[i, 1]
            q_y = quat[i, 2]
            q_z = quat[i, 3]
            
            q_ww = q_w * q_w
            q_wx = q_w * q_x
            q_wy = q_w * q_y
            q_wz = q_w * q_z
            q_xx = q_x * q_x
            q_xy = q_x * q_y
            q_xz = q_x * q_z
            q_yy = q_y * q_y
            q_yz = q_y * q_z
            q_zz = q_z * q_z
            
            denom = q_ww + q_xx + q_yy + q_zz
            
            out[i, 0] = (v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy)) / denom
            out[i, 1] = (v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx)) / denom
            out[i, 2] = (v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz)) / denom
    
    @ti.kernel
    def _compose_quat_kernel(self, quat1: ti.types.ndarray(), quat2: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for composing two quaternions."""
        w1 = quat1[0]
        x1 = quat1[1]
        y1 = quat1[2]
        z1 = quat1[3]
        
        w2 = quat2[0]
        x2 = quat2[1]
        y2 = quat2[2]
        z2 = quat2[3]
        
        out[0] = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
        out[1] = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
        out[2] = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
        out[3] = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
    
    @ti.kernel
    def _compose_quat_batch_kernel(self, quat1: ti.types.ndarray(), quat2: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for composing two quaternion arrays."""
        for i in range(quat1.shape[0]):
            w1 = quat1[i, 0]
            x1 = quat1[i, 1]
            y1 = quat1[i, 2]
            z1 = quat1[i, 3]
            
            w2 = quat2[i, 0]
            x2 = quat2[i, 1]
            y2 = quat2[i, 2]
            z2 = quat2[i, 3]
            
            out[i, 0] = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
            out[i, 1] = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
            out[i, 2] = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
            out[i, 3] = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
    
    @ti.kernel
    def _rotation_vector_to_quat_kernel(self, rot_vecs: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for converting rotation vectors to quaternions."""
        for i in range(rot_vecs.shape[0]):
            vx = rot_vecs[i, 0]
            vy = rot_vecs[i, 1]
            vz = rot_vecs[i, 2]
            
            angle = ti.sqrt(vx * vx + vy * vy + vz * vz)
            
            if angle < self.EPS:
                # Zero rotation -> identity quaternion
                out[i, 0] = 1.0
                out[i, 1] = 0.0
                out[i, 2] = 0.0
                out[i, 3] = 0.0
            else:
                half_angle = angle / 2.0
                sin_half = ti.sin(half_angle)
                cos_half = ti.cos(half_angle)
                
                # Normalize axis
                axis_x = vx / angle
                axis_y = vy / angle
                axis_z = vz / angle
                
                out[i, 0] = cos_half
                out[i, 1] = sin_half * axis_x
                out[i, 2] = sin_half * axis_y
                out[i, 3] = sin_half * axis_z
    
    @ti.kernel
    def _quat_to_rotation_matrix_kernel(self, quat: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for converting quaternions to rotation matrices."""
        for i in range(quat.shape[0]):
            w = quat[i, 0]
            x = quat[i, 1]
            y = quat[i, 2]
            z = quat[i, 3]
            
            x2 = x * x
            y2 = y * y
            z2 = z * z
            xy = x * y
            xz = x * z
            yz = y * z
            wx = w * x
            wy = w * y
            wz = w * z
            
            out[i, 0, 0] = 1.0 - 2.0 * (y2 + z2)
            out[i, 0, 1] = 2.0 * (xy - wz)
            out[i, 0, 2] = 2.0 * (xz + wy)
            out[i, 1, 0] = 2.0 * (xy + wz)
            out[i, 1, 1] = 1.0 - 2.0 * (x2 + z2)
            out[i, 1, 2] = 2.0 * (yz - wx)
            out[i, 2, 0] = 2.0 * (xz - wy)
            out[i, 2, 1] = 2.0 * (yz + wx)
            out[i, 2, 2] = 1.0 - 2.0 * (x2 + y2)
    
    @ti.kernel
    def _matrix_multiply_kernel(self, a: ti.types.ndarray(), b: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for matrix multiplication (2D)."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = 0.0
                for k in range(a.shape[1]):
                    out[i, j] += a[i, k] * b[k, j]
    
    @ti.kernel
    def _matrix_multiply_batch_kernel(self, a: ti.types.ndarray(), b: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for batched matrix multiplication (3D)."""
        for i in range(a.shape[0]):
            for j in range(out.shape[1]):
                for k in range(out.shape[2]):
                    out[i, j, k] = 0.0
                    for l in range(a.shape[2]):
                        out[i, j, k] += a[i, j, l] * b[i, l, k]
    
    @ti.kernel
    def _sum_axis_0_kernel(self, arr: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for summing along axis 0."""
        for j in range(arr.shape[1]):
            out[j] = 0.0
            for i in range(arr.shape[0]):
                out[j] += arr[i, j]
    
    @ti.kernel
    def _triu_kernel(self, m: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for upper triangular part."""
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if j >= i:
                    out[i, j] = m[i, j]
                else:
                    out[i, j] = 0.0
    
    @ti.kernel
    def _negate_kernel_1d(self, arr: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for negating 1D array."""
        for i in range(arr.shape[0]):
            out[i] = -arr[i]
    
    @ti.kernel
    def _negate_kernel_2d(self, arr: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for negating 2D array."""
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                out[i, j] = -arr[i, j]
    
    @ti.kernel
    def _diag_kernel(self, arr: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for creating diagonal matrix from 1D array."""
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if i == j:
                    out[i, j] = arr[i]
                else:
                    out[i, j] = 0.0
    
    @ti.kernel
    def _add_diag_kernel(self, m: ti.types.ndarray(), diag: ti.types.ndarray()):
        """Taichi kernel for adding diagonal to matrix."""
        for i in range(m.shape[0]):
            m[i, i] += diag[i]
    
    @ti.kernel
    def _subtract_matrix_kernel(self, m: ti.types.ndarray(), sub: ti.types.ndarray()):
        """Taichi kernel for subtracting matrix from matrix."""
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                m[i, j] -= sub[i, j]
    
    @ti.kernel
    def _elementwise_ops_kernel(self, a: ti.types.ndarray(), b: ti.types.ndarray(), c: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for element-wise operations: out = a * (b - c)."""
        for i in range(out.shape[0]):
            out[i] = a[i] * (b[i] - c[i])
    
    @ti.kernel
    def _add_scaled_kernel(self, a: ti.types.ndarray(), b: ti.types.ndarray(), scale: ti.f64, out: ti.types.ndarray()):
        """Taichi kernel for: out = a + b * scale."""
        for i in range(out.shape[0]):
            out[i] = a[i] + b[i] * scale
    
    @ti.kernel
    def _vstack_kernel_2d(self, first: ti.types.ndarray(), rest: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for vstacking arrays."""
        # Copy first row
        for j in range(first.shape[0]):
            out[0, j] = first[j]
        # Copy rest
        for i in range(rest.shape[0]):
            for j in range(rest.shape[1]):
                out[i + 1, j] = rest[i, j]

    def transform_by_quat(self, vecs, quats):
        """Apply quaternion rotation to vectors."""
        # Handle single vector/quaternion case
        vecs_was_1d = len(vecs.shape) == 1
        quats_was_1d = len(quats.shape) == 1
        
        if vecs_was_1d:
            vecs_reshaped = ti.ndarray(dtype=ti.f64, shape=(1, vecs.shape[0]))
            for i in range(vecs.shape[0]):
                vecs_reshaped[0, i] = vecs[i]
            vecs = vecs_reshaped
        if quats_was_1d:
            quats_reshaped = ti.ndarray(dtype=ti.f64, shape=(1, quats.shape[0]))
            for i in range(quats.shape[0]):
                quats_reshaped[0, i] = quats[i]
            quats = quats_reshaped
        
        # Normalize quaternions
        quat_norms = ti.ndarray(dtype=ti.f64, shape=(quats.shape[0],))
        quats_normalized = ti.ndarray(dtype=ti.f64, shape=quats.shape)
        self._normalize_quat_kernel(quats, quat_norms, quats_normalized)
        
        out = ti.ndarray(dtype=ti.f64, shape=vecs.shape)
        self._transform_by_quat_kernel(vecs, quats_normalized, out)
        
        # Restore original shape if needed
        if vecs_was_1d and quats_was_1d:
            result = ti.ndarray(dtype=ti.f64, shape=(out.shape[1],))
            for i in range(out.shape[1]):
                result[i] = out[0, i]
            return result
        
        return out

    def compose_quat_by_quat(self, quat2, quat1):
        """Compose two quaternions."""
        # Ensure 1D shape
        if len(quat1.shape) > 1:
            quat1_1d = ti.ndarray(dtype=ti.f64, shape=(4,))
            for i in range(4):
                quat1_1d[i] = quat1[0, i] if quat1.shape[0] == 1 else quat1[i, 0]
            quat1 = quat1_1d
        if len(quat2.shape) > 1:
            quat2_1d = ti.ndarray(dtype=ti.f64, shape=(4,))
            for i in range(4):
                quat2_1d[i] = quat2[0, i] if quat2.shape[0] == 1 else quat2[i, 0]
            quat2 = quat2_1d
        
        result = ti.ndarray(dtype=ti.f64, shape=(4,))
        self._compose_quat_kernel(quat1, quat2, result)
        return result

    def compose_quat_by_quat_batch(self, quat2, quat1):
        """Compose two quaternion arrays."""
        # Ensure 2D shape (batch_size, 4)
        if len(quat1.shape) == 1:
            quat1_2d = ti.ndarray(dtype=ti.f64, shape=(1, 4))
            for i in range(4):
                quat1_2d[0, i] = quat1[i]
            quat1 = quat1_2d
        if len(quat2.shape) == 1:
            quat2_2d = ti.ndarray(dtype=ti.f64, shape=(1, 4))
            for i in range(4):
                quat2_2d[0, i] = quat2[i]
            quat2 = quat2_2d
        
        result = ti.ndarray(dtype=ti.f64, shape=quat1.shape)
        self._compose_quat_batch_kernel(quat1, quat2, result)
        return result

    def rotation_vector_to_quat(self, rotation_vectors):
        """Convert rotation vectors to quaternions."""
        # Handle single vector case
        single_vec = len(rotation_vectors.shape) == 1
        if single_vec:
            rot_vecs_2d = ti.ndarray(dtype=ti.f64, shape=(1, 3))
            for i in range(3):
                rot_vecs_2d[0, i] = rotation_vectors[i]
            rotation_vectors = rot_vecs_2d
        
        quat = ti.ndarray(dtype=ti.f64, shape=(rotation_vectors.shape[0], 4))
        self._rotation_vector_to_quat_kernel(rotation_vectors, quat)
        
        if single_vec:
            quat_1d = ti.ndarray(dtype=ti.f64, shape=(4,))
            for i in range(4):
                quat_1d[i] = quat[0, i]
            return quat_1d
        
        return quat

    def quat_to_rotation_matrix(self, quat):
        """Convert quaternions to rotation matrices."""
        # Handle single quaternion case
        single_quat = len(quat.shape) == 1
        if single_quat:
            quat_2d = ti.ndarray(dtype=ti.f64, shape=(1, 4))
            for i in range(4):
                quat_2d[0, i] = quat[i]
            quat = quat_2d
        
        # Normalize quaternions
        quat_norms = ti.ndarray(dtype=ti.f64, shape=(quat.shape[0],))
        quat_normalized = ti.ndarray(dtype=ti.f64, shape=quat.shape)
        self._normalize_quat_kernel(quat, quat_norms, quat_normalized)
        
        # Convert to rotation matrices
        R = ti.ndarray(dtype=ti.f64, shape=(quat_normalized.shape[0], 3, 3))
        self._quat_to_rotation_matrix_kernel(quat_normalized, R)
        
        if single_quat:
            R_2d = ti.ndarray(dtype=ti.f64, shape=(3, 3))
            for i in range(3):
                for j in range(3):
                    R_2d[i, j] = R[0, i, j]
            return R_2d
        
        return R

    # ------------------------- forward-kinematics utils ----------------------
    @ti.kernel
    def _copy_array_2d_kernel(self, src: ti.types.ndarray(), dst: ti.types.ndarray()):
        """Taichi kernel for copying 2D array."""
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                dst[i, j] = src[i, j]
    
    @ti.kernel
    def _compute_link_quat_pos_kernel(self, pos: ti.types.ndarray(), joint_axis: ti.types.ndarray(),
                                      link_initial_quat: ti.types.ndarray(), link_initial_pos: ti.types.ndarray(),
                                      link_quat: ti.types.ndarray(), link_pos: ti.types.ndarray(),
                                      link_quat0: ti.types.ndarray(), link_rotation_vector_quat: ti.types.ndarray()):
        """Taichi kernel for computing link quaternions and positions."""
        dofs = link_quat.shape[0]
        
        # Compute rotation vectors
        for i in range(dofs):
            axis_x = pos[i] * joint_axis[i, 0]
            axis_y = pos[i] * joint_axis[i, 1]
            axis_z = pos[i] * joint_axis[i, 2]
            
            # Convert to quaternion
            angle = ti.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
            if angle < self.EPS:
                link_rotation_vector_quat[i, 0] = 1.0
                link_rotation_vector_quat[i, 1] = 0.0
                link_rotation_vector_quat[i, 2] = 0.0
                link_rotation_vector_quat[i, 3] = 0.0
            else:
                half_angle = angle / 2.0
                sin_half = ti.sin(half_angle)
                cos_half = ti.cos(half_angle)
                axis_x_norm = axis_x / angle
                axis_y_norm = axis_y / angle
                axis_z_norm = axis_z / angle
                link_rotation_vector_quat[i, 0] = cos_half
                link_rotation_vector_quat[i, 1] = sin_half * axis_x_norm
                link_rotation_vector_quat[i, 2] = sin_half * axis_y_norm
                link_rotation_vector_quat[i, 3] = sin_half * axis_z_norm
        
        # Initialize link_quat0
        for i in range(dofs):
            for j in range(4):
                link_quat0[i, j] = link_initial_quat[i, j]
        
        # Process each link sequentially since computations depend on i-1
        ti.loop_config(serialize=True)
        for i in range(dofs):
            if i == 0:
                # First link: just copy initial position
                for j in range(3):
                    link_pos[i, j] = link_initial_pos[i, j]
            else:
                # Compose: use previous link quaternion first, then current initial quat
                qp_w = link_quat[i-1, 0]
                qp_x = link_quat[i-1, 1]
                qp_y = link_quat[i-1, 2]
                qp_z = link_quat[i-1, 3]
                
                q0_w = link_quat0[i, 0]
                q0_x = link_quat0[i, 1]
                q0_y = link_quat0[i, 2]
                q0_z = link_quat0[i, 3]
                
                link_quat0[i, 0] = qp_w * q0_w - qp_x * q0_x - qp_y * q0_y - qp_z * q0_z
                link_quat0[i, 1] = qp_w * q0_x + qp_x * q0_w + qp_y * q0_z - qp_z * q0_y
                link_quat0[i, 2] = qp_w * q0_y - qp_x * q0_z + qp_y * q0_w + qp_z * q0_x
                link_quat0[i, 3] = qp_w * q0_z + qp_x * q0_y - qp_y * q0_x + qp_z * q0_w
                
                # Transform relative position by previous quaternion
                v_x = link_initial_pos[i, 0]
                v_y = link_initial_pos[i, 1]
                v_z = link_initial_pos[i, 2]
                
                q_ww = qp_w * qp_w
                q_wx = qp_w * qp_x
                q_wy = qp_w * qp_y
                q_wz = qp_w * qp_z
                q_xx = qp_x * qp_x
                q_xy = qp_x * qp_y
                q_xz = qp_x * qp_z
                q_yy = qp_y * qp_y
                q_yz = qp_y * qp_z
                q_zz = qp_z * qp_z
                denom = q_ww + q_xx + q_yy + q_zz
                
                rel_pos_x = (v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy)) / denom
                rel_pos_y = (v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx)) / denom
                rel_pos_z = (v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz)) / denom
                
                # Add to previous position
                link_pos[i, 0] = rel_pos_x + link_pos[i-1, 0]
                link_pos[i, 1] = rel_pos_y + link_pos[i-1, 1]
                link_pos[i, 2] = rel_pos_z + link_pos[i-1, 2]
            
            # Compose final quaternion: link_quat = rotation_quat * quat0
            rq_w = link_rotation_vector_quat[i, 0]
            rq_x = link_rotation_vector_quat[i, 1]
            rq_y = link_rotation_vector_quat[i, 2]
            rq_z = link_rotation_vector_quat[i, 3]
            
            q0_w = link_quat0[i, 0]
            q0_x = link_quat0[i, 1]
            q0_y = link_quat0[i, 2]
            q0_z = link_quat0[i, 3]
            
            w = q0_w * rq_w - q0_x * rq_x - q0_y * rq_y - q0_z * rq_z
            x = q0_w * rq_x + q0_x * rq_w + q0_y * rq_z - q0_z * rq_y
            y = q0_w * rq_y - q0_x * rq_z + q0_y * rq_w + q0_z * rq_x
            z = q0_w * rq_z + q0_x * rq_y - q0_y * rq_x + q0_z * rq_w
            
            link_quat[i, 0] = w
            link_quat[i, 1] = x
            link_quat[i, 2] = y
            link_quat[i, 3] = z
    
    def compute_link_quat_pos(self, pos):
        """Compute link quaternions and positions."""
        dofs = self.config.dofs
        # Initialize as Taichi arrays
        link_quat = ti.ndarray(dtype=ti.f64, shape=(dofs, Configuration.NUM_DIMS_QUAT))
        link_pos = ti.ndarray(dtype=ti.f64, shape=(dofs, Configuration.NUM_DIMS_3D))
        link_quat0 = ti.ndarray(dtype=ti.f64, shape=self.config_state.link_initial_quat_no_base.shape)
        link_rotation_vector_quat = ti.ndarray(dtype=ti.f64, shape=(dofs, Configuration.NUM_DIMS_QUAT))
        
        self._compute_link_quat_pos_kernel(pos, self.config_state.joint_axis,
                                          self.config_state.link_initial_quat_no_base,
                                          self.config_state.link_initial_pos_no_base,
                                          link_quat, link_pos, link_quat0, link_rotation_vector_quat)

        return link_quat, link_pos, link_quat0, link_pos, link_rotation_vector_quat

    def compute_xaxis(self, joint_axis, link_quat0):
        """Compute x-axis."""
        return self.transform_by_quat(joint_axis, link_quat0)

    def compute_linear_and_angular_jacobian(self, xaxis, COM, xanchor):
        """Compute linear and angular jacobians."""
        angular_jacobian = xaxis
        COM_matrix = self.tile_row(COM)
        linear_jacobian = self.cross_product(xaxis, self._taichi_subtract(COM_matrix, xanchor))
        return angular_jacobian, linear_jacobian

    def forward_kinematics(self, pos0):
        """Compute forward kinematics."""
        link_quat, link_pos, link_quat0, link_pos0, link_rotation_vector_quat = self.compute_link_quat_pos(pos0)

        xanchor = link_pos0

        joint_axis = self.config_state.joint_axis
        xaxis = self.compute_xaxis(joint_axis, link_quat0)

        angular_jacobian = xaxis

        COM = self.compute_COM(link_quat, link_pos)

        angular_jacobian, linear_jacobian = self.compute_linear_and_angular_jacobian(xaxis, COM, xanchor)

        return angular_jacobian, linear_jacobian, link_quat, link_pos, COM

    def forward_dynamics(self, pos0, vel0, linear_jacobian, angular_jacobian, link_quat, link_pos, COM):
        """Compute forward dynamics."""
        link_cinr_inertial, link_cinr_pos, link_inertial_pos = self.compute_link_inertia(link_quat, link_pos, COM)

        f2_ang, f2_vel, link_angular_vel, link_linear_vel, link_angular_vel_individual, link_linear_vel_individual, f2_vel_vel, f2_ang_vel = self.compute_f2(link_cinr_inertial, link_cinr_pos, linear_jacobian, angular_jacobian, vel0)

        joint_linear_jacobian_acc, joint_angular_jacobian_acc = self.compute_joint_jacobian_acc(link_angular_vel, link_linear_vel, linear_jacobian, angular_jacobian)

        f1_vel, f1_ang, link_linear_acc, link_angular_acc, link_linear_acc_individual, link_angular_acc_individual = self.compute_f1(link_cinr_inertial, link_cinr_pos, joint_linear_jacobian_acc, joint_angular_jacobian_acc, vel0)

        link_force, link_torque = self.compute_link_force_torque(f1_vel, f1_ang, f2_vel, f2_ang)

        bias_force, bias_force_angular, bias_force_linear = self.compute_bias_force(link_torque, link_force, angular_jacobian, linear_jacobian)

        control_force, applied_force = self.compute_applied_force(pos0, vel0)

        force = self.compute_force(bias_force, applied_force)

        return force, link_cinr_pos, link_cinr_inertial

    def mass(self, link_cinr_pos, link_cinr_inertial, angular_jacobian, linear_jacobian):
        """Compute mass matrix."""
        crb_pos, crb_inertial, crb_mass = self.compute_crb(link_cinr_pos, link_cinr_inertial)

        f_ang, f_vel = self.compute_f_ang_vel(crb_pos, crb_inertial, crb_mass, angular_jacobian, linear_jacobian)

        mass_matrix = self.compute_mass_matrix(f_ang, f_vel, angular_jacobian, linear_jacobian)

        return mass_matrix

    def step(self):
        """Perform one simulation step."""
        # Swap previous and current entity so the next call uses the newly computed values
        self.previous_entity, self.current_entity = self.current_entity, self.previous_entity

        pos0 = self.previous_entity.joint.pos
        vel0 = self.previous_entity.joint.vel

        angular_jacobian, linear_jacobian, link_quat, link_pos, COM = self.forward_kinematics(pos0)

        # Store link state in current_entity
        self.current_entity.link.quat = link_quat
        self.current_entity.link.pos = link_pos

        force, link_cinr_pos, link_cinr_inertial = self.forward_dynamics(pos0, vel0,
                                                                        linear_jacobian, angular_jacobian,
                                                                        link_quat, link_pos,
                                                                        COM)

        mass_matrix = self.mass(link_cinr_pos, link_cinr_inertial, angular_jacobian, linear_jacobian)

        acc, vel, pos = self.compute_newton_euler(mass_matrix, force, pos0, vel0)

        # Store results in current_entity
        self.current_entity.joint.pos = pos
        self.current_entity.joint.vel = vel

    @ti.kernel
    def _extract_row_kernel(self, src: ti.types.ndarray(), row_idx: int, dst: ti.types.ndarray()):
        """Taichi kernel for extracting a row from 2D array."""
        for j in range(src.shape[1]):
            dst[j] = src[row_idx, j]
    
    def get_pos(self):
        """Get current position."""
        return self.current_entity.joint.pos

    def get_vel(self):
        """Get current velocity."""
        return self.current_entity.joint.vel

    def set_pos(self, pos):
        """Set current position."""
        # If pos is None or not initialized, create new array
        if self.current_entity.joint.pos is None:
            self.current_entity.joint.pos = ti.ndarray(dtype=ti.f64, shape=pos.shape)
        # Copy data to avoid aliasing
        self._list_to_ndarray_kernel_1d(pos, self.current_entity.joint.pos)

    def set_vel(self, vel):
        """Set current velocity."""
        # If vel is None or not initialized, create new array
        if self.current_entity.joint.vel is None:
            self.current_entity.joint.vel = ti.ndarray(dtype=ti.f64, shape=vel.shape)
        # Copy data to avoid aliasing
        self._list_to_ndarray_kernel_1d(vel, self.current_entity.joint.vel)

    def get_link_quat(self, link_name = None):
        """Get current link quaternion."""
        link_quat = self.current_entity.link.quat
        if link_name is not None:
            link_id = self.config.link_ids[link_name]
            # Extract row using kernel
            result = ti.ndarray(dtype=ti.f64, shape=(link_quat.shape[1],))
            self._extract_row_kernel(link_quat, link_id, result)
            return result
        # Return full array
        return link_quat

    def get_link_pos(self, link_name = None):
        """Get current link position."""
        link_pos = self.current_entity.link.pos
        if link_name is not None:
            link_id = self.config.link_ids[link_name]
            # Extract row using kernel
            result = ti.ndarray(dtype=ti.f64, shape=(link_pos.shape[1],))
            self._extract_row_kernel(link_pos, link_id, result)
            return result
        return link_pos

    def control_dofs_position(self, pos):
        """Control the position of the DOFs."""
        self.config_state.control_pos = pos

    def compute_joint_jacobian_acc(self, link_angular_vel, link_linear_vel, linear_jacobian, angular_jacobian):
        """Compute joint jacobian acceleration."""
        link_angular_vel_shifted = self.shift_bottom(link_angular_vel)
        link_linear_vel_shifted = self.shift_bottom(link_linear_vel)
        joint_linear_jacobian_acc = self._taichi_add(self.cross_product(link_angular_vel_shifted, linear_jacobian), self.cross_product(link_linear_vel_shifted, angular_jacobian))
        joint_angular_jacobian_acc = self.cross_product(link_angular_vel_shifted, angular_jacobian)
        return joint_linear_jacobian_acc, joint_angular_jacobian_acc

    def compute_f1(self, link_cinr_inertia, link_cinr_pos, joint_linear_jacobian_acc, joint_angular_jacobian_acc, vel0):
        """Compute f1 terms."""
        link_linear_acc_individual = self.multiply_scalar_by_vector(vel0, joint_linear_jacobian_acc)
        gravity = self.config_state.gravity
        link_linear_acc = self._taichi_add(gravity, self.cumulative_sum(link_linear_acc_individual))

        link_angular_acc_individual = self.multiply_scalar_by_vector(vel0, joint_angular_jacobian_acc)
        link_angular_acc = self.cumulative_sum(link_angular_acc_individual)

        f1_ang = self._taichi_add(self.multiply_matrix_by_vector(link_cinr_inertia, link_angular_acc), self.cross_product(link_cinr_pos, link_linear_acc))

        link_mass = self.config_state.link_mass_no_base
        f1_vel = self._taichi_subtract(self.multiply_scalar_by_vector(link_mass, link_linear_acc), self.cross_product(link_cinr_pos, link_angular_acc))

        return f1_vel, f1_ang, link_linear_acc, link_angular_acc, link_linear_acc_individual, link_angular_acc_individual

    def compute_f2(self, link_inertia, link_cinr_pos, linear_jacobian, angular_jacobian, vel0):
        """Compute f2 terms."""
        link_linear_vel_individual = self.multiply_scalar_by_vector(vel0, linear_jacobian)
        link_linear_vel = self.cumulative_sum(link_linear_vel_individual)
        link_angular_vel_individual = self.multiply_scalar_by_vector(vel0, angular_jacobian)
        link_angular_vel = self.cumulative_sum(link_angular_vel_individual)
        link_mass = self.config_state.link_mass_no_base
        f2_vel_vel = self._taichi_subtract(self.multiply_scalar_by_vector(link_mass, link_linear_vel), self.cross_product(link_cinr_pos, link_angular_vel))
        f2_vel = self.cross_product(link_angular_vel, f2_vel_vel)
        f2_ang_vel = self._taichi_add(self.multiply_matrix_by_vector(link_inertia, link_angular_vel), self.cross_product(link_cinr_pos, link_linear_vel))
        f2_ang = self._taichi_add(self.cross_product(link_angular_vel, f2_ang_vel), self.cross_product(link_linear_vel, f2_vel_vel))
        return f2_ang, f2_vel, link_angular_vel, link_linear_vel, link_angular_vel_individual, link_linear_vel_individual, f2_vel_vel, f2_ang_vel

    def compute_link_force_torque(self, f1_vel, f1_ang, f2_vel, f2_ang):
        """Compute link force and torque."""
        link_force_individual = self._taichi_add(f1_vel, f2_vel)
        link_torque_individual = self._taichi_add(f1_ang, f2_ang)

        # traverse the kinematic chain in reverse to get the cumulative forces/torques bottom-up
        link_force = self.reverse_cumulative_sum(link_force_individual)
        link_torque = self.reverse_cumulative_sum(link_torque_individual)
        return link_force, link_torque

    @ti.kernel
    def _compute_link_inertia_kernel(self, rotation: ti.types.ndarray(), link_inertia: ti.types.ndarray(),
                                     link_cinr_inertial: ti.types.ndarray()):
        """Taichi kernel for computing link inertia with rotation."""
        n = rotation.shape[0]
        for i in range(n):
            # Compute rotation @ link_inertia @ rotation.T
            # First: temp = rotation @ link_inertia
            for j in range(3):
                for k in range(3):
                    temp = 0.0
                    for l in range(3):
                        temp += rotation[i, j, l] * link_inertia[i, l, k]
                    link_cinr_inertial[i, j, k] = temp
            
            # Then: result = temp @ rotation.T
            for j in range(3):
                for k in range(3):
                    temp = 0.0
                    for l in range(3):
                        temp += link_cinr_inertial[i, j, l] * rotation[i, k, l]
                    link_cinr_inertial[i, j, k] = temp
    
    def compute_link_inertia(self, link_quat, link_pos, COM):
        """Compute link inertia."""
        link_inertial_quat = self.compose_quat_by_quat_batch(link_quat, self.config_state.link_inertial_quat_no_base)

        rotation = self.quat_to_rotation_matrix(link_inertial_quat)  # Returns Taichi ndarray
        
        link_cinr_inertial = ti.ndarray(dtype=ti.f64, shape=self.config_state.link_inertia_no_base.shape)
        self._compute_link_inertia_kernel(rotation, self.config_state.link_inertia_no_base, link_cinr_inertial)

        link_inertial_pos = self.transform_by_quat(self.config_state.link_inertial_pos_no_base, link_quat)
        link_inertial_pos = self._taichi_add(link_inertial_pos, link_pos)
        link_inertial_pos = self._taichi_subtract(link_inertial_pos, COM)

        link_cinr_pos = self.multiply_scalar_by_vector(self.config_state.link_mass_no_base, link_inertial_pos)

        hhT = self.hhT_batch(link_inertial_pos)
        hhT_mass = self.multiply_scalar_by_matrix(self.config_state.link_mass_no_base, hhT)
        link_cinr_inertial = self._taichi_add(link_cinr_inertial, hhT_mass)

        return link_cinr_inertial, link_cinr_pos, link_inertial_pos

    def compute_force(self, bias_force, applied_force):
        """Compute force."""
        # Negate bias_force
        neg_bias_force = ti.ndarray(dtype=ti.f64, shape=bias_force.shape)
        if len(bias_force.shape) == 1:
            self._negate_kernel_1d(bias_force, neg_bias_force)
        else:
            self._negate_kernel_2d(bias_force, neg_bias_force)
        
        return self._taichi_add(neg_bias_force, applied_force)

    def compute_bias_force(self, link_torque, link_force, angular_jacobian, linear_jacobian):
        """Compute bias force."""
        bias_force_angular = self.scalar_product(angular_jacobian, link_torque)
        bias_force_linear = self.scalar_product(linear_jacobian, link_force)
        return self._taichi_add(bias_force_angular, bias_force_linear), bias_force_angular, bias_force_linear

    @ti.kernel
    def _compute_control_force_kernel(self, Kp: ti.types.ndarray(), Kv: ti.types.ndarray(),
                                       control_pos: ti.types.ndarray(), pos0: ti.types.ndarray(),
                                       vel0: ti.types.ndarray(), out: ti.types.ndarray()):
        """Taichi kernel for computing control force."""
        for i in range(out.shape[0]):
            out[i] = Kp[i] * (control_pos[i] - pos0[i]) - Kv[i] * vel0[i]
    
    def compute_applied_force(self, pos0, vel0):
        """Compute applied force."""
        Kp = self.config_state.Kp
        Kv = self.config_state.Kv
        control_pos = self.config_state.control_pos
        min_force = self.config_state.min_force
        max_force = self.config_state.max_force
        
        control_force = ti.ndarray(dtype=ti.f64, shape=pos0.shape)
        self._compute_control_force_kernel(Kp, Kv, control_pos, pos0, vel0, control_force)
        applied_force = self.clip(control_force, min_force, max_force)
        return control_force, applied_force

    @ti.kernel
    def _compute_newton_euler_kernel(self, acc: ti.types.ndarray(), vel0: ti.types.ndarray(),
                                      pos0: ti.types.ndarray(), step_dt: ti.f64,
                                      vel: ti.types.ndarray(), pos: ti.types.ndarray()):
        """Taichi kernel for Newton-Euler step."""
        for i in range(vel.shape[0]):
            vel[i] = vel0[i] + acc[i] * step_dt
            pos[i] = pos0[i] + vel[i] * step_dt
    
    def compute_newton_euler(self, mass, force, pos0, vel0):
        """Compute Newton-Euler step."""
        acc = self.linalg_solve(mass, force)
        step_dt = self.config.step_dt
        
        vel = ti.ndarray(dtype=ti.f64, shape=vel0.shape)
        pos = ti.ndarray(dtype=ti.f64, shape=pos0.shape)
        self._compute_newton_euler_kernel(acc, vel0, pos0, step_dt, vel, pos)
        return acc, vel, pos

    @ti.kernel
    def _compute_com_kernel(self, link_quat_full: ti.types.ndarray(), link_pos_full: ti.types.ndarray(),
                            link_inertial_pos_full: ti.types.ndarray(), link_mass_full: ti.types.ndarray(),
                            com: ti.types.ndarray()):
        """Taichi kernel for computing center of mass."""
        n_links = link_quat_full.shape[0]
        
        # Compute weighted positions
        weighted_sum = ti.Vector([0.0, 0.0, 0.0])
        mass_sum = 0.0
        
        for i in range(n_links):
            # Transform inertial position by quaternion
            v_x = link_inertial_pos_full[i, 0]
            v_y = link_inertial_pos_full[i, 1]
            v_z = link_inertial_pos_full[i, 2]
            
            q_w = link_quat_full[i, 0]
            q_x = link_quat_full[i, 1]
            q_y = link_quat_full[i, 2]
            q_z = link_quat_full[i, 3]
            
            q_ww = q_w * q_w
            q_wx = q_w * q_x
            q_wy = q_w * q_y
            q_wz = q_w * q_z
            q_xx = q_x * q_x
            q_xy = q_x * q_y
            q_xz = q_x * q_z
            q_yy = q_y * q_y
            q_yz = q_y * q_z
            q_zz = q_z * q_z
            denom = q_ww + q_xx + q_yy + q_zz
            
            i_pos_x = (v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy)) / denom
            i_pos_y = (v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx)) / denom
            i_pos_z = (v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz)) / denom
            
            # Add link position
            pos_x = i_pos_x + link_pos_full[i, 0]
            pos_y = i_pos_y + link_pos_full[i, 1]
            pos_z = i_pos_z + link_pos_full[i, 2]
            
            # Weight by mass
            mass = link_mass_full[i]
            weighted_sum[0] += pos_x * mass
            weighted_sum[1] += pos_y * mass
            weighted_sum[2] += pos_z * mass
            mass_sum += mass
        
        # Compute COM
        if mass_sum > 0.0:
            com[0] = weighted_sum[0] / mass_sum
            com[1] = weighted_sum[1] / mass_sum
            com[2] = weighted_sum[2] / mass_sum
        else:
            com[0] = 0.0
            com[1] = 0.0
            com[2] = 0.0
    
    @ti.kernel
    def _prepend_base_link_kernel(self, link_quat: ti.types.ndarray(), link_pos: ti.types.ndarray(),
                                   link_quat_full: ti.types.ndarray(), link_pos_full: ti.types.ndarray()):
        """Taichi kernel for prepending base link."""
        # Set base link
        link_quat_full[0, 0] = 1.0
        link_quat_full[0, 1] = 0.0
        link_quat_full[0, 2] = 0.0
        link_quat_full[0, 3] = 0.0
        link_pos_full[0, 0] = 0.0
        link_pos_full[0, 1] = 0.0
        link_pos_full[0, 2] = 0.0
        
        # Copy rest
        for i in range(link_quat.shape[0]):
            for j in range(4):
                link_quat_full[i + 1, j] = link_quat[i, j]
            for j in range(3):
                link_pos_full[i + 1, j] = link_pos[i, j]
    
    def compute_COM(self, link_quat, link_pos):
        """Compute center of mass."""
        # Prepend base link quaternion and position
        dofs = link_quat.shape[0]
        link_quat_full = ti.ndarray(dtype=ti.f64, shape=(dofs + 1, 4))
        link_pos_full = ti.ndarray(dtype=ti.f64, shape=(dofs + 1, 3))
        
        self._prepend_base_link_kernel(link_quat, link_pos, link_quat_full, link_pos_full)
        
        # Use full versions from config_state
        link_inertial_pos_full = self.config_state.link_inertial_pos
        link_mass_full = self.config_state.link_mass
        
        com = ti.ndarray(dtype=ti.f64, shape=(3,))
        self._compute_com_kernel(link_quat_full, link_pos_full, link_inertial_pos_full, link_mass_full, com)
        return com

    def compute_f_ang_vel(self, expected_crb_pos, expected_crb_inertial, expected_crb_mass,
                                expected_angular_jacobian, expected_linear_jacobian):
        """Compute f_ang and f_vel."""
        expected_f_ang = self._taichi_add(self.multiply_matrix_by_vector(expected_crb_inertial, expected_angular_jacobian), self.cross_product(expected_crb_pos, expected_linear_jacobian))
        expected_f_vel = self._taichi_subtract(self.multiply_scalar_by_vector(expected_crb_mass, expected_linear_jacobian), self.cross_product(expected_crb_pos, expected_angular_jacobian))
        return expected_f_ang, expected_f_vel

    @ti.kernel
    def _compute_mass_matrix_kernel(self, f_ang: ti.types.ndarray(), f_vel: ti.types.ndarray(),
                                    ang_jac: ti.types.ndarray(), lin_jac: ti.types.ndarray(),
                                    armature: ti.types.ndarray(), Kv: ti.types.ndarray(),
                                    step_dt: ti.f64, mass_matrix: ti.types.ndarray()):
        """Taichi kernel for computing mass matrix.
        
        Computes: mass_matrix = f_ang @ ang_jac.T + f_vel @ lin_jac.T
        Where f_ang and f_vel are (dofs, 3) and ang_jac/lin_jac are (dofs, 3)
        Result is (dofs, dofs) where result[i, j] = sum over d of f_ang[i, d] * ang_jac[j, d] + f_vel[i, d] * lin_jac[j, d]
        """
        dofs = mass_matrix.shape[0]
        dims = f_ang.shape[1]  # Should be 3 for 3D
        
        # Compute f_ang @ ang_jac.T + f_vel @ lin_jac.T
        # This is: result[i, j] = sum over d of f_ang[i, d] * ang_jac[j, d] + f_vel[i, d] * lin_jac[j, d]
        for i in range(dofs):
            for j in range(dofs):
                mass_matrix[i, j] = 0.0
                for d in range(dims):
                    mass_matrix[i, j] += f_ang[i, d] * ang_jac[j, d] + f_vel[i, d] * lin_jac[j, d]
        
        # Update lower triangular part with upper triangular part
        for i in range(dofs):
            for j in range(dofs):
                if j < i:
                    mass_matrix[i, j] = mass_matrix[j, i]
        
        # Add armature
        for i in range(dofs):
            mass_matrix[i, i] += armature[i]
        
        # Subtract force jacobian for implicit integration
        # Note: force_jacobian = -diag(Kv), so subtracting it means adding step_dt * Kv
        # This matches PyTorch: mass_matrix -= step_dt * (-diag(Kv)) = mass_matrix += step_dt * diag(Kv)
        for i in range(dofs):
            mass_matrix[i, i] += step_dt * Kv[i]
    
    def compute_mass_matrix(self, expected_f_ang, expected_f_vel, angular_jacobian, linear_jacobian):
        """Compute mass matrix."""
        dofs = angular_jacobian.shape[0]  # Number of DOFs (links), not the 3D dimension
        mass_matrix = ti.ndarray(dtype=ti.f64, shape=(dofs, dofs))
        
        self._compute_mass_matrix_kernel(expected_f_ang, expected_f_vel, angular_jacobian, linear_jacobian,
                                        self.config_state.armature, self.config_state.Kv,
                                        self.config.step_dt, mass_matrix)
        return mass_matrix

    def compute_crb(self, expected_cinr_pos, expected_cinr_inertial):
        """Compute composite rigid body."""
        expected_crb_pos = self.reverse_cumulative_sum(expected_cinr_pos)
        expected_crb_inertial = self.reverse_cumulative_sum(expected_cinr_inertial)

        link_mass = self.config_state.link_mass_no_base
        expected_crb_mass = self.reverse_cumulative_sum(link_mass)

        return expected_crb_pos, expected_crb_inertial, expected_crb_mass

