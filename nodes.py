""" Classes for different types of nodes """
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from attributes import DecisionNodeAttributes, AttributeType, LeafNodeAttributes
from node_functions import continuous_test_fn, get_distribution
from checking import check_attributes


class Node(ABC):
    """ Class implementing a generic node of a decision tree """
    @abstractmethod
    def get_level(self) -> int:
        """ Returns the level of the node in the tree """

    @abstractmethod
    def get_label(self) -> str:
        """ Returns the name of the node """

    @abstractmethod
    def get_parent_node(self) -> Node:
        """ Get the node's parent """


class DecisionNode(Node):
    """ Class implementing a decision node of a decision tree """
    @abstractmethod
    def get_attribute(self) -> str:
        """ Returns the attribute of the node """

    @abstractmethod
    def add_child(self, child):
        """ Adds another node to the set of node childs """

    @abstractmethod
    def delete_child(self, child):
        """ Removes a node from the set of node childs """

    @abstractmethod
    def get_children(self) -> set:
        """ Returns the set of the node childs """

    @abstractmethod
    def get_child(self, attr_value: float) -> Node:
        """ Returns the child corresponding to the attribute """


class DecisionNodeContinuous(DecisionNode):
    """ Decision node splitting data on continuous attribute """
    def __init__(self, attributes: DecisionNodeAttributes, parent_node: Node):
        check_attributes(attributes, parent_node)
        self._parent_node = parent_node
        self._childs = set()
        self._attributes = attributes

    def get_level(self) -> int:
        """ Returns the level of the node in the tree """
        return self._attributes.node_level

    def get_label(self) -> str:
        """ Returns the name of the node """
        return self._attributes.node_name

    def get_parent_node(self) -> Node:
        """ Get the node's parent """
        return self._parent_node

    def get_attribute(self) -> str:
        """ Returns the attribute of the node """
        return self._attributes.attribute_name

    def add_child(self, child):
        """ Adds another node to the set of node childs """
        self._childs.add(child)

    def delete_child(self, child):
        """ Removes a node from the set of node childs """
        self._childs.remove(child)

    def get_children(self) -> set:
        """ Returns the set of the node childs """
        return self._childs

    def _run_continuous_test(self, attr_value):
        """ checks weather the value is below the threshold """
        return continuous_test_fn(self._attributes.threshold, attr_value)

    def run_test(self, attr_value: float) -> str:
        """ runs the test on attr_value and returns the test as a string """
        if self._run_continuous_test(float(attr_value)):
            return f"{self._attributes.attribute_name} <= {self._attributes.threshold}"
        return f"{self._attributes.attribute_name} > {self._attributes.threshold}"

    def get_child(self, attr_value: float) -> Node:
        """ Returns the child compatible with the attribute value """
        required_child = None
        for child in self.get_children():
            if child.get_label() == self.run_test(attr_value):
                required_child = child
        return required_child


class DecisionNodeCategorical(DecisionNode):
    """ Decision node splitting data on categorical attribute """
    def __init__(self, attributes: DecisionNodeAttributes, parent_node: Node):
        #check_attributes(attributes, parent_node)
        self._parent_node = parent_node
        self._children = set()
        self._attributes = attributes

    def get_level(self) -> int:
        """ Returns the level of the node in the tree """
        return self._attributes.node_level

    def get_label(self) -> str:
        """ Returns the name of the node """
        return self._attributes.node_name

    def get_parent_node(self) -> Node:
        """ Get the node's parent """
        return self._parent_node

    def get_attribute(self) -> str:
        """ Returns the attribute of the node """
        return self._attributes.attribute_name

    def add_child(self, child):
        """ Adds another node to the set of node childs """
        self._children.add(child)

    def delete_child(self, child):
        """ Removes a node from the set of node childs """
        self._children.remove(child)

    def get_children(self) -> set:
        """ Returns the set of the node childs """
        return self._children

    def run_test(self, attr_value: str | bool) -> str:
        """ runs the test on attr_value and returns the test as a string """
        return f"{self._attributes.attribute_name} = {attr_value}"

    def get_child(self, attr_value: str | bool) -> Node:
        """ Returns the child fulfilling the condition given by the attribute value """
        required_child = None
        for child in self.get_children():
            if child.get_label() == self.run_test(attr_value):
                required_child = child
        return required_child


class LeafNode(Node):
    """ class implementing a leaf node of the decision tree """
    def __init__(self, attributes: LeafNodeAttributes, parent_node: Node):
        #self._check_attributes(attributes, parent_node)
        self._parent_node = parent_node
        self._childs = set()
        self._attributes = attributes
        # The label class of the node is the class with maximum number of examples

    def get_level(self) -> int:
        """ Returns the level of the node in the tree """
        return self._attributes.node_level

    def get_label(self) -> str:
        """ Returns the name of the node """
        return self._attributes.node_name

    def get_parent_node(self) -> Node:
        """ Get the node's parent """
        return self._parent_node

    def _get_class_name(self) -> str:
        """ get the class with maximum number of samples """
        return max(zip(self._attributes.classes.values(), self._attributes.classes.keys()))[1]

    def _get_classes_distribution(self) -> dict:
        """ Returns the distribution over the classes inside the leaf """
        return get_distribution(self._attributes.classes)

    def get_purity(self) -> float:
        """ returns the percentage of the class with more samples (purity) """
        return np.round(self._get_classes_distribution()[self._get_class_name()], 4)

    def get_classes(self) -> dict:
        """ return the classes dictionary """
        return self._attributes.classes

    def get_classes_names(self) ->dict:
        """ return the names of the target classes """
        return list(self._attributes.classes.keys())

    def get_instances_number(self) -> int:
        """ returns the number of samples classified in the leaf """
        return sum(self._attributes.classes.values())
#    def get_class_name(self) -> list:
#        """ Returns the classes with maximum number """
#        return self._attributes.classes.get_class_name()
#
#    def get_classes(self) -> dict:
#        """ Returns the classes contained in the node after the training """
#        return self._attributes.classes.get_classes()
#
#    def get_classes_distribution(self) -> dict:
#        """ Returns the number of examples of a class contained in the node after the training """
#        return self._attributes.classes.get_classes_distribution()
#
#    def get_leaf_purity(self) -> float:
#        """ returns the purity of the targets distribution """
#        return self._attributes.classes.get_purity()
#
#    def get_classes_frequency_dictionary(self) -> dict:
#        """
#        returns a dictionary containining for each class its frequency and the total
#        number of samples in the leaf
#        """
#        return self._attributes.classes.classes_frequency_dictionary()
#
#    def get_total_sum(self) -> int:
#        """ returns the total number of samples in the leaf """
#        return self._attributes.classes.get_total_sum()
