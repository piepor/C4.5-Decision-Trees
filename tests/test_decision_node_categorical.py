import pytest
from attributes import AttributeType, NodeAttributes
from nodes import DecisionNodeCategorical

@pytest.fixture
def root_attributes():
    return NodeAttributes(0, "root", "A", AttributeType.CATEGORICAL, None)

@pytest.fixture
def node_a_attributes():
    return NodeAttributes(1, "A = Red", "B", AttributeType.CATEGORICAL, None)

@pytest.fixture
def node_b_attributes():
    return NodeAttributes(1, "A = Blue", "C", AttributeType.CATEGORICAL, None)

@pytest.fixture
def node_c_attributes():
    return NodeAttributes(1, "A = Green", "D", AttributeType.CATEGORICAL, None)

def test_wrong_input_type(root_attributes):
    """ test exception with wrong input """
    root_node = DecisionNodeCategorical(root_attributes, None)
    with pytest.raises(TypeError):
        test_result = root_node.run_test(10)

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

def test_does_not_get_child(root_attributes, node_a_attributes, node_b_attributes, node_c_attributes):
    node_b_attributes.node_name = "A = Red"
    node_c_attributes.node_name = "A = Red"
    root_node = DecisionNodeCategorical(root_attributes, None)
    node_a = DecisionNodeCategorical(node_a_attributes, root_node)
    node_b = DecisionNodeCategorical(node_b_attributes, root_node)
    node_c = DecisionNodeCategorical(node_c_attributes, root_node)
    root_node.add_child(node_a)
    root_node.add_child(node_b)
    root_node.add_child(node_c)
    with pytest.raises(RuntimeError):
        root_node.get_child("Blue")
