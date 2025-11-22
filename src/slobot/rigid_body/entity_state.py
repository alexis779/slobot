from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import get_origin, Union

import numpy as np
import torch


class VectorType(Enum):
    """Enum for specifying how to convert lists to vector types."""
    RAW_LIST = "raw_list"
    NUMPY_ARRAY = "numpy_array"
    TORCH_TENSOR = "torch_tensor"


def from_dict(klass, data, vector_type: VectorType):
    """Recursively deserialize a raw object into a dataclass instance.
    
    Args:
        klass: The class type to deserialize into
        data: The raw data (dict, list, etc.) to deserialize
        vector_type: How to convert lists to vector types
    
    Returns:
        Deserialized object of type klass
    """

    if type(data) is list:
        # Assume data is a list of raw floats, convert to appropriate vector type
        match vector_type:
            case VectorType.NUMPY_ARRAY:
                return np.array(data)
            case VectorType.TORCH_TENSOR:
                return torch.tensor(data)
            case VectorType.RAW_LIST:
                return data
    
    # Check if klass is a dataclass
    if is_dataclass(klass):
        # Recursively deserialize each field
        return klass(**{field.name: from_dict(field.type, data[field.name], vector_type) for field in fields(klass)})
    
    # return the data as-is
    return data

@dataclass
class JointState:
    """Joint state containing position and velocity."""
    
    pos: Union[np.ndarray, torch.Tensor, list[float], None]
    vel: Union[np.ndarray, torch.Tensor, list[float], None]


@dataclass
class LinkState:
    """Link state containing position and orientation."""
    
    pos: Union[np.ndarray, torch.Tensor, list[float], None]
    quat: Union[np.ndarray, torch.Tensor, list[float], None]


@dataclass
class EntityState:
    """Entity state containing joint and link states."""
    
    joint: JointState
    link: LinkState


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