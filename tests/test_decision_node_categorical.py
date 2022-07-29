import pytest
from attributes import AttributeType, DecisionNodeAttributes, NodeType
from nodes import DecisionNodeCategorical

@pytest.fixture
def root_attributes():
    return DecisionNodeAttributes(0, "root", NodeType.DECISION_NODE_CATEGORICAL,
            "A", AttributeType.CATEGORICAL, None)

@pytest.fixture
def node_a_attributes():
    return DecisionNodeAttributes(1, "A = Red", NodeType.DECISION_NODE_CATEGORICAL,
            "B", AttributeType.CATEGORICAL, None)

@pytest.fixture
def node_b_attributes():
    return DecisionNodeAttributes(1, "A = Blue", NodeType.DECISION_NODE_CATEGORICAL,
            "C", AttributeType.CATEGORICAL, None)

@pytest.fixture
def node_c_attributes():
    return DecisionNodeAttributes(1, "A = Green", NodeType.DECISION_NODE_CATEGORICAL,
            "D", AttributeType.CATEGORICAL, None)

def test_run_test(root_attributes):
    """ test the returning string """
    root_node = DecisionNodeCategorical(root_attributes, None)
    test_result = "A = Blue"
    assert test_result == root_node.run_test("Blue")

def test_get_child(root_attributes, node_a_attributes, node_b_attributes, node_c_attributes):
    """ must return the node corrsponding to the attribute value """
    root_node = DecisionNodeCategorical(root_attributes, None)
    node_a = DecisionNodeCategorical(node_a_attributes, root_node)
    node_b = DecisionNodeCategorical(node_b_attributes, root_node)
    node_c = DecisionNodeCategorical(node_c_attributes, root_node)
    root_node.add_child(node_a)
    root_node.add_child(node_b)
    root_node.add_child(node_c)
    assert root_node.get_child("Red") == node_a

def test_adding_child(root_attributes, node_a_attributes):
    """ add a child """
    root_node = DecisionNodeCategorical(root_attributes, None)
    node_a = DecisionNodeCategorical(node_a_attributes, root_node)
    root_node.add_child(node_a)
    assert root_node.get_children() == set([node_a])

def test_adding_childs(root_attributes, node_a_attributes, node_b_attributes):
    """ add two childs """
    root_node = DecisionNodeCategorical(root_attributes, None)
    node_a = DecisionNodeCategorical(node_a_attributes, root_node)
    node_b = DecisionNodeCategorical(node_b_attributes, root_node)
    root_node.add_child(node_a)
    root_node.add_child(node_b)
    assert root_node.get_children() == set([node_a, node_b])

def test_delete_child(root_attributes, node_a_attributes, node_b_attributes):
    """ delete a child """
    root_node = DecisionNodeCategorical(root_attributes, None)
    node_a = DecisionNodeCategorical(node_b_attributes, root_node)
    node_b = DecisionNodeCategorical(node_b_attributes, root_node)
    root_node.add_child(node_a)
    root_node.add_child(node_b)
    root_node.delete_child(node_a)
    assert root_node.get_children() == set([node_b])
