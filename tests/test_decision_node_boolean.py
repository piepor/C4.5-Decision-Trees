import pytest
from attributes import AttributeType, DecisionNodeAttributes
from nodes import DecisionNodeBoolean

@pytest.fixture
def root_attributes():
    return DecisionNodeAttributes(0, "root", "A", AttributeType.BOOLEAN, None)

@pytest.fixture
def node_a_attributes():
    return DecisionNodeAttributes(1, "A = True", "B", AttributeType.BOOLEAN, None)

@pytest.fixture
def node_b_attributes():
    return DecisionNodeAttributes(1, "A = False", "C", AttributeType.BOOLEAN, None)

def test_wrong_input_type(root_attributes):
    """ test exception with wrong input """
    root_node = DecisionNodeBoolean(root_attributes, None)
    with pytest.raises(TypeError):
        test_result = root_node.run_test(10)

def test_run_test(root_attributes):
    """ test the returning string """
    root_node = DecisionNodeBoolean(root_attributes, None)
    test_result = "A = False"
    assert test_result == root_node.run_test(False)

def test_get_child(root_attributes, node_a_attributes, node_b_attributes):
    """ must return the node corrsponding to the attribute value """
    root_node = DecisionNodeBoolean(root_attributes, None)
    node_a = DecisionNodeBoolean(node_a_attributes, root_node)
    node_b = DecisionNodeBoolean(node_b_attributes, root_node)
    root_node.add_child(node_a)
    root_node.add_child(node_b)
    assert root_node.get_child(False) == node_b

def test_does_not_get_child(root_attributes, node_a_attributes, node_b_attributes):
    node_b_attributes.node_name = "A = True"
    root_node = DecisionNodeBoolean(root_attributes, None)
    node_a = DecisionNodeBoolean(node_a_attributes, root_node)
    node_b = DecisionNodeBoolean(node_b_attributes, root_node)
    root_node.add_child(node_a)
    root_node.add_child(node_b)
    with pytest.raises(RuntimeError):
        root_node.get_child(False)
