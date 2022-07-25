""" Defines a class containing information about the node attributes needed in the init """
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Callable
from leaf_classes import LeafClasses


class AttributeType(Enum):
    """ types of attributes considered in the tree """
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class DecisionNodeAttributes:
    """ attributes class """
    node_level: int
    node_name: str
    attribute_name: str
    attribute_type: AttributeType
    threshold: float = None

@dataclass
class LeafNodeAttributes:
    """ attributes class """
    node_level: int
    node_name: str
    classes: LeafClasses
