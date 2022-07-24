""" Classes for different types of nodes """
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
from attributes import NodeAttributes, AttributeType

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
            raise Exception("A node can't have itself as child.")
        self._childs.add(child)

    def delete_child(self, child):
        """ Removes a node from the set of node childs """
        self._childs.remove(child)

    def get_childs(self) -> set:
        """ Returns the set of the node childs """
        return self._childs


class DecisionNodeContinuous(DecisionNode):
    """ Decision node splitting data on continuous attribute """
#    def __init__(self, attributes: NodeAttributes, parent_node: Node):
#        super().__init__(attributes, parent_node)

    def _run_continuous_test(self, attr_value):
        """ checks weather the value is below the threshold """
        return self._attributes.test_fn(self._attributes.threshold, attr_value)

    def run_test(self, attr_value: float) -> str:
        """ runs the test on attr_value and returns the test as a string """
        if self._run_continuous_test(attr_value):
            return f"{self._attributes.attribute_name} <= {self._attributes.threshold}"
        return f"{self._attributes.attribute_name} > {self._attributes.threshold}"
        # name of the node. Corresponds to the condition to be fulfilled in the parent node

#    def get_child(self, attr_value) -> Node:
#        """ Returns the child fulfilling the condition given by the attribute value """
#        required_child = None
#        for child in self.get_childs():
#            if child.get_label() == self.run_test(attr_value):
#                required_child = child
#        return required_child

#    def delete_child(self, child: Node):
#        """ Removes a node from the set of node childs """
#        self._childs.remove(child)



#    def _run_continuous_test(self, attr_value: float) -> str:
#        """ runs the test for continuous values """
#        attr_name = self._attribute.split(':')[0]
#        if self.continuous_test_function_lower(attr_value):
#            result_test = f"{attr_name} <= {self._threshold}"
#        else:
#            result_test = f"{attr_name} > {self._threshold}"
#        return result_test
#
#    def run_test(self, attr_value: str) -> str:
#        """ Returns the condition given by the attribute value """
#        if self._ == 'continuous':
#            result_test = self._run_continuous_test(float(attr_value))
#        else:
#            attr_name = self._attribute.split(':')[0]
#            result_test = f"{attr_name} = {attr_value}"
#        return result_test
#
#    def continuous_test_function_lower(self, attr_value) -> bool:
#        """ Test if the attribute value is less than the threshold """
#        return attr_value <= self._threshold


#        self._label: str = attributes.node_name
#        # attribute the node splits on.
#        # For continuous attributes the format is: "attribute:threshold"
#        self._attribute: str = attributes.attribute_name
#        # type of the attribute ['categorical', 'boolean', 'continuous']
#        self._attribute_type: AttributeType = attributes.attribute_type
#        self._parent_node: Node = attributes.parent_node
#        self._level: str = attributes.parent_level + 1
#        self._threshold: float = attributes.threshold
#        self._test_fn: Callable[float, float] = attributes.test_fn
#class Node(ABC):
#    """ Class implementing a generic node of a decision tree """
#
#    def __init__(self, parent_attribute_value: str, parent_level: int):
#        # name of the node. Corresponds to the condition to be fulfilled in the parent node
#        self._label = parent_attribute_value
#        self._childs = set()
#        # attribute the node splits on.
#        # For continuous attributes the format is: "attribute:threshold"
#        self._attribute = None
#        # type of the attribute ['categorical', 'boolean', 'continuous']
#        self._attribute_type = None
#        self._parent_node = None
#        self._level = parent_level + 1
#        self._threshold = None
#
#    def set_attribute_continuous(self, attribute: str):
#        """ Sets the attribute (and its type) on which the node splits """
#        self._attribute = attribute
#        self._attribute_type = 'continuous'
#        self._threshold = float(self._attribute.split(':')[1])
#
#    def set_attribute_categorical(self, attribute: str):
#        """ Sets the attribute (and its type) on which the node splits """
#        self._attribute = attribute
#        self._attribute_type = 'categorical'
#
#    def set_attribute_boolean(self, attribute: str):
#        """ Sets the attribute (and its type) on which the node splits """
#        self._attribute = attribute
#        self._attribute_type = 'boolean'
#
#    def set_parent_node(self, parent_node) -> None:
#        """ Set the node's parent """
#        self._parent_node = parent_node
#
#    def get_level(self) -> int:
#        """ Returns the level of the node in the tree """
#        return self._level
#
#    def get_label(self) -> str:
#        """ Returns the name of the node """
#        return self._label
#
#    def get_attribute(self) -> str:
#        """ Returns the attribute of the node """
#        return self._attribute
#
#    def get_parent_node(self) -> Node:
#        """ Get the node's parent """
#        return self._parent_node


#class DecisionNode(Node):
#    """ Class implementing a decision node of a decision tree """
#
#    def get_childs(self) -> set:
#        """ Returns the set of the node childs """
#        return self._childs
#
#    def get_child(self, attr_value) -> Node:
#        """ Returns the child fulfilling the condition given by the attribute value """
#        required_child = None
#        for child in self.get_childs():
#            if child.get_label() == self.run_test(attr_value):
#                required_child = child
#        return required_child
#
#    def delete_child(self, child: Node):
#        """ Removes a node from the set of node childs """
#        self._childs.remove(child)
#
#    def _run_continuous_test(self, attr_value: float) -> str:
#        """ runs the test for continuous values """
#        attr_name = self._attribute.split(':')[0]
#        if self.continuous_test_function_lower(attr_value):
#            result_test = f"{attr_name} <= {self._threshold}"
#        else:
#            result_test = f"{attr_name} > {self._threshold}"
#        return result_test
#
#    def run_test(self, attr_value: str) -> str:
#        """ Returns the condition given by the attribute value """
#        if self._attribute_type == 'continuous':
#            result_test = self._run_continuous_test(float(attr_value))
#        else:
#            attr_name = self._attribute.split(':')[0]
#            result_test = f"{attr_name} = {attr_value}"
#        return result_test
#
#    def continuous_test_function_lower(self, attr_value) -> bool:
#        """ Test if the attribute value is less than the threshold """
#        return attr_value <= self._threshold
