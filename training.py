from nodes import DecisionNodeBoolean, DecisionNodeCategorical
from nodes import DecisionNodeContinuous, LeafNode, Node
from attributes import NodeAttributes


def create_continuous_decision_node(node_attributes: NodeAttributes, parent_node: Node) -> None:
    """ create a continuous decision node """
    return DecisionNodeContinuous(node_attributes, parent_node)


def create_categorical_decision_node(node_attributes: NodeAttributes, parent_node: Node) -> None:
    """ create a continuous decision node """
    return DecisionNodeCategorical(node_attributes, parent_node)


def create_boolean_decision_node(node_attributes: NodeAttributes, parent_node: Node) -> None:
    """ create a continuous decision node """
    return DecisionNodeBoolean(node_attributes, parent_node)


def create_leaf_node(node_attributes: NodeAttributes, parent_node: Node) -> None:
    """ create a continuous decision node """
    return LeafNode(node_attributes, parent_node)
