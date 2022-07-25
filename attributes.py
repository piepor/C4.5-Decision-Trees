""" Defines a class containing information about the node attributes needed in the init """
from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable
from leaf_classes import LeafClasses


class NodeType(Enum):
    """ types of node in the decision tree """
    DECISION_NODE_CONTINUOUS = auto()
    DECISION_NODE_CATEGORICAL = auto()
    DECISION_NODE_BOOLEAN = auto()
    LEAF_NODE = auto()


class AttributeType(Enum):
    """ types of attributes considered in the tree """
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class NodeAttributes:
    """ general node attributes """
    node_level: int
    node_name: str
    node_type: NodeType


@dataclass
class DecisionNodeAttributes(NodeAttributes):
    """ decision node attributes class """
    attribute_name: str
    attribute_type: AttributeType
    threshold: float = None


@dataclass
class LeafNodeAttributes(NodeAttributes):
    """ leaf attributes class """
    classes: LeafClasses


@dataclass
class DecisionTreeAttributes:
    """ decision tree attributes class """
    attributes_map: dict
    max_depth: int = 20
    node_purity: float = 0.9
    max_instances: int = 20


def from_str_to_enum(attributes_map: dict) -> dict:
    """ convert a string into a value of type AttributeType """
    for name in attributes_map:
        if attributes_map[name] == "continuous":
            attributes_map[name] = AttributeType.CONTINUOUS
        elif attributes_map[name] == "categorical":
            attributes_map[name] = AttributeType.CATEGORICAL
        elif attributes_map[name] == "boolean":
            attributes_map[name] = AttributeType.CATEGORICAL
        else:
            raise TypeError("Attribute type not supported")
    return attributes_map
