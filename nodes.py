""" Classes for different types of nodes """
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
from attributes import NodeAttributes, AttributeType
from predicting import continuous_test_fn

class Node(ABC):
    """ Class implementing a generic node of a decision tree """
    @abstractmethod
    def get_level(self) -> int:
        """ Returns the level of the node in the tree """

    @abstractmethod
    def get_label(self) -> str:
        """ Returns the name of the node """

    @abstractmethod
    def get_attribute(self) -> str:
        """ Returns the attribute of the node """

    @abstractmethod
    def get_parent_node(self) -> Node:
        """ Get the node's parent """


class DecisionNode(Node):
    """ Class implementing a decision node of a decision tree """
    def __init__(self, attributes: NodeAttributes, parent_node: Node):
        self._check_attributes(attributes, parent_node)
        self._parent_node = parent_node
        self._childs = set()
        self._attributes = attributes

    def _check_attributes(self, attributes, parent_node):
        if attributes.node_name == "root" and not attributes.node_level == 0:
            raise ValueError("Root node must have level 0")
        if attributes.node_level < 0:
            raise ValueError(f"Level must be positive. \
                    Node [{attributes.node_name}] - Parent [{parent_node.get_label()}")
        if attributes.attribute_type == AttributeType.CONTINUOUS and not attributes.threshold:
            parent_name = None
            if parent_node:
                parent_name = parent_node.get_label() # the root node does not have a parent node
            raise ValueError(f"A continuous type must have a threshold. \
                    Node [{attributes.node_name}] - Parent [{parent_name}")
        if not attributes.attribute_type == AttributeType.CONTINUOUS and attributes.threshold:
            parent_name = None
            if parent_node:
                parent_name = parent_node.get_label() # the root node does not have a parent node
            raise ValueError(f"A threshold has been set on non continuous node type. \
                    Node [{attributes.node_name}] - Parent [{parent_name}")

    def get_level(self) -> int:
        """ Returns the level of the node in the tree """
        return self._attributes.node_level

    def get_label(self) -> str:
        """ Returns the name of the node """
        return self._attributes.node_name

    def get_attribute(self) -> str:
        """ Returns the attribute of the node """
        return self._attributes.attribute_name

    def get_parent_node(self) -> Node:
        """ Get the node's parent """
        return self._parent_node

    def add_child(self, child):
        """ Adds another node to the set of node childs """
        if child == self:
            raise ValueError("A node can't have itself as child.")
        self._childs.add(child)

    def delete_child(self, child):
        """ Removes a node from the set of node childs """
        self._childs.remove(child)

    def get_childs(self) -> set:
        """ Returns the set of the node childs """
        return self._childs


class DecisionNodeContinuous(DecisionNode):
    """ Decision node splitting data on continuous attribute """
    def _run_continuous_test(self, attr_value):
        """ checks weather the value is below the threshold """
        return continuous_test_fn(self._attributes.threshold, attr_value)

    def run_test(self, attr_value: float) -> str:
        """ runs the test on attr_value and returns the test as a string """
        if not isinstance(attr_value, float):
            raise TypeError("A continuous decision node's input must be a float \
                    Node [{attributes.node_name}] - Parent [{parent_name}")
        if self._run_continuous_test(float(attr_value)):
            return f"{self._attributes.attribute_name} <= {self._attributes.threshold}"
        return f"{self._attributes.attribute_name} > {self._attributes.threshold}"

    def get_child(self, attr_value: float) -> Node:
        """ Returns the child compatible with the attribute value """
        required_child = None
        for child in self.get_childs():
            if child.get_label() == self.run_test(attr_value):
                required_child = child
        if not required_child:
            raise RuntimeError(
            "Can't find a child compatible with {self._attributes.attribute_name} \
            {attr_value}. Node [{attributes.node_name}] - Parent [{parent_name}")
        return required_child

class DecisionNodeBoolean(DecisionNode):
    """ Decision node splitting data on boolean attribute """
    def run_test(self, attr_value: bool) -> str:
        """ runs the test on attr_value and returns the test as a string """
        if not isinstance(attr_value, bool):
            raise TypeError("A boolean decision node's input must be a bool \
                    Node [{attributes.node_name}] - Parent [{parent_name}")
        return f"{self._attributes.attribute_name} = {attr_value}"

    def get_child(self, attr_value: str) -> Node:
        """ Returns the child fulfilling the condition given by the attribute value """
        required_child = None
        for child in self.get_childs():
            if child.get_label() == self.run_test(attr_value):
                required_child = child
        if not required_child:
            raise RuntimeError(
            "Can't find a child compatible with {self._attributes.attribute_name} \
            {attr_value}. Node [{attributes.node_name}] - Parent [{parent_name}")
        return required_child

class DecisionNodeCategorical(DecisionNode):
    """ Decision node splitting data on categorical attribute """
    def run_test(self, attr_value: str) -> str:
        """ runs the test on attr_value and returns the test as a string """
        if not isinstance(attr_value, str):
            raise TypeError("A categorical decision node's input must be a bool \
                    Node [{attributes.node_name}] - Parent [{parent_name}")
        return f"{self._attributes.attribute_name} = {attr_value}"

    def get_child(self, attr_value: str) -> Node:
        """ Returns the child fulfilling the condition given by the attribute value """
        required_child = None
        for child in self.get_childs():
            if child.get_label() == self.run_test(attr_value):
                required_child = child
        if not required_child:
            raise RuntimeError(
            "Can't find a child compatible with {self._attributes.attribute_name} \
            {attr_value}. Node [{attributes.node_name}] - Parent [{parent_name}")
        return required_child
