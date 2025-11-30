import json
from dataclasses import dataclass, fields, is_dataclass
from typing import Callable, TypeAlias, Union


VectorLike: TypeAlias = Union[list, None] # np.ndarray, torch.Tensor, ti.ndarray, 


VectorFactory: TypeAlias = Callable[[list], VectorLike]


def from_dict(klass, data, vector_factory: VectorFactory):
    """Recursively deserialize a raw object into a dataclass instance.
    
    Args:
        klass: The class type to deserialize into
        data: The raw data (dict, list, etc.) to deserialize
        vector_factory: Callable that converts Python lists into vector objects
    
    Returns:
        Deserialized object of type klass
    """

    if type(data) is list:
        # Assume data is a list of raw floats, convert using provided factory
        return vector_factory(data)
    
    # Check if klass is a dataclass
    if is_dataclass(klass):
        # Recursively instantiate each field
        return klass(**{field.name: from_dict(field.type, data[field.name], vector_factory) for field in fields(klass)})
    
    # return the data as-is
    return data

@dataclass
class JointState:
    """Joint state containing position and velocity."""
    
    pos: VectorLike
    vel: VectorLike


@dataclass
class LinkState:
    """Link state containing position and orientation."""
    
    pos: VectorLike
    quat: VectorLike


@dataclass
class ConfigurationState:
    """Configuration state containing all configuration parameters as vectors."""
    
    gravity: VectorLike
    middle_pos_offset: VectorLike
    min_force: VectorLike
    max_force: VectorLike
    min_dofs_limit: VectorLike
    max_dofs_limit: VectorLike
    Kp: VectorLike
    Kv: VectorLike
    control_pos: VectorLike
    joint_axis: VectorLike
    link_initial_quat: VectorLike  # With base link
    link_initial_pos: VectorLike  # With base link
    link_mass: VectorLike  # With base link
    link_inertia: VectorLike  # With base link
    link_inertial_quat: VectorLike  # With base link
    link_inertial_pos: VectorLike  # With base link
    armature: VectorLike
    # Versions without base link (first element/row excluded)
    link_initial_quat_no_base: VectorLike = None
    link_initial_pos_no_base: VectorLike = None
    link_mass_no_base: VectorLike = None
    link_inertia_no_base: VectorLike = None
    link_inertial_quat_no_base: VectorLike = None
    link_inertial_pos_no_base: VectorLike = None


@dataclass
class OptimizerParametersState:
    """Serializable set of optimizer tunable parameters."""
    
    middle_pos_offset: VectorLike
    min_force: VectorLike
    max_force: VectorLike
    Kp: VectorLike
    Kv: VectorLike
    armature: VectorLike


@dataclass
class EntityState:
    """Entity state containing joint and link states."""
    
    joint: JointState
    link: LinkState


def get_state_values(state_obj):
    """Return the field values for a given state object in declaration order."""
    return [getattr(state_obj, field.name) for field in fields(state_obj)]

def to_dict(state_obj):
    """Convert a state object to a dictionary."""
    return {field.name: getattr(state_obj, field.name).tolist() for field in fields(state_obj)}


def load_attributes(state_obj, destination_obj):
    """Load the attributes of state_obj into destination_obj."""
    for field in fields(state_obj):
        setattr(destination_obj, field.name, getattr(state_obj, field.name))

def create_entity_state():
    """Factory function to create an EntityState with fields initialized to None.
    
    Returns:
        EntityState with all fields initialized to None:
        - joint.pos: None
        - joint.vel: None
        - link.pos: None
        - link.quat: None
    """
    joint = JointState(pos=None, vel=None)
    link = LinkState(pos=None, quat=None)
    
    return EntityState(joint=joint, link=link)


DEFAULT_STEPS_CSV_PATH = "./tests/steps.csv"


def load_csv_rows(vector_factory: VectorFactory, csv_path: str = DEFAULT_STEPS_CSV_PATH):
    """Load entity states from a JSON-per-line CSV file.
    
    Args:
        vector_factory: Callable that converts Python lists into vector objects.
        csv_path: Path to the CSV file containing JSON rows.
    
    Returns:
        List of EntityState objects deserialized from the CSV file.
    """
    entity_states: list[EntityState] = []
    
    with open(csv_path) as file_obj:
        for line in file_obj:
            raw_data = json.loads(line)
            entity_state = from_dict(EntityState, raw_data, vector_factory)
            entity_states.append(entity_state)
    
    return entity_states

