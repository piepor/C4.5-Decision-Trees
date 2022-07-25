from decision_tree_refactor import DecisionTree
from attributes import DecisionTreeAttributes, DecisionNodeAttributes
from attributes import NodeType, AttributeType
from nodes import DecisionNodeContinuous
import pytest


@pytest.fixture
def decision_tree():
    attributes_map = {"attr1": "categorical"}
    decision_tree_attributes = DecisionTreeAttributes(attributes_map)
    return DecisionTree(decision_tree_attributes)


@pytest.fixture
def root_attributes():
    return DecisionNodeAttributes(0, "root", NodeType.DECISION_NODE_CONTINUOUS,
            "A", AttributeType.CONTINUOUS, 10.0)


@pytest.fixture
def node_a_attributes():
    return DecisionNodeAttributes(1, "A <= 10.0", NodeType.DECISION_NODE_CONTINUOUS,
            "B", AttributeType.CONTINUOUS, 10.0)


def test_add_node(decision_tree, root_attributes):
    node = decision_tree.create_node(root_attributes, None)
    decision_tree.add_node(node)
    assert node in decision_tree.get_nodes()


def test_delete_node(decision_tree, root_attributes, node_a_attributes):
    root_node = decision_tree.create_node(root_attributes, None)
    decision_tree.add_node(root_node)
    node_a = decision_tree.create_node(node_a_attributes, root_node)
    decision_tree.add_node(node_a)
    decision_tree.delete_node(node_a)
    assert root_node in decision_tree.get_nodes()
    assert node_a not in decision_tree.get_nodes()
    assert node_a not in root_node.get_childs()
