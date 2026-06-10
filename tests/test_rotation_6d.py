import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from slobot.hilserl.ee_kinematics import (
    EE_STATE_DIM,
    quat_wxyz_to_rotation_6d,
    rotation_6d_to_matrix,
    rotation_6d_to_quat_wxyz,
    rotation_matrix_to_6d,
)
from slobot.hilserl.models.ee_state import EeAction, EeObservation
from slobot.hilserl.models.gripper_pose import GripperLinkPose


class TestRotation6d(unittest.TestCase):
    def test_roundtrip_random_rotations(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            quat_xyzw = R.random(random_state=rng).as_quat()
            quat_wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
            recovered = rotation_6d_to_quat_wxyz(quat_wxyz_to_rotation_6d(quat_wxyz))
            dot = abs(np.dot(np.array(quat_wxyz), np.array(recovered)))
            self.assertAlmostEqual(dot, 1.0, places=5)

    def test_matrix_roundtrip(self):
        rotation = R.from_euler("xyz", [0.3, -1.1, 2.0]).as_matrix()
        rotation_6d = rotation_matrix_to_6d(rotation)
        recovered = rotation_6d_to_matrix(rotation_6d)
        np.testing.assert_allclose(recovered, rotation, atol=1e-6)

    def test_ee_state_tensor_roundtrip(self):
        pose = GripperLinkPose(
            position=(0.1, -0.2, 0.3),
            rotation_6d=quat_wxyz_to_rotation_6d((0.9, 0.1, 0.2, 0.3)),
        )
        ee_action = EeAction(pose=pose, jaw_rad=0.4)
        tensor = ee_action.to_tensor()
        self.assertEqual(tensor.numel(), EE_STATE_DIM)
        recovered = EeAction.from_tensor(tensor)
        np.testing.assert_allclose(recovered.pose.position, pose.position, atol=1e-5)
        np.testing.assert_allclose(recovered.pose.rotation_6d, pose.rotation_6d, atol=1e-5)
        self.assertAlmostEqual(recovered.jaw_rad, ee_action.jaw_rad, places=5)

        ee_obs = EeObservation(pose=pose, jaw_rad=0.4)
        self.assertEqual(ee_obs.to_tensor().numel(), EE_STATE_DIM)


if __name__ == "__main__":
    unittest.main()
