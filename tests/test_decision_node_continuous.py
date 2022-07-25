from nodes import DecisionNodeContinuous
from attributes import DecisionNodeAttributes, AttributeType, NodeType
import pytest

@pytest.fixture
def root_attributes():
    return DecisionNodeAttributes(0, "root", NodeType.DECISION_NODE_CONTINUOUS,
            "A", AttributeType.CONTINUOUS, 10.0)

@pytest.fixture
def root_attributes_negative():
    return DecisionNodeAttributes(0, "root", NodeType.DECISION_NODE_CONTINUOUS,
            "A", AttributeType.CONTINUOUS, -10.0)

@pytest.fixture
def node_a_attributes():
    return DecisionNodeAttributes(1, "A <= 10.0", NodeType.DECISION_NODE_CONTINUOUS,
            "B", AttributeType.CONTINUOUS, 10.0)

@pytest.fixture
def node_b_attributes():
    return DecisionNodeAttributes(1, "A > 10.0", NodeType.DECISION_NODE_CONTINUOUS,
            "C", AttributeType.CONTINUOUS, 10.0)

def test_wrong_input_type(root_attributes):
    root_node = DecisionNodeContinuous(root_attributes, None)
    with pytest.raises(TypeError):
        test_result = root_node.run_test(True)

def test_dnc_below_threshold(root_attributes):
    """ running test with attribute below the threshold """
    root_node = DecisionNodeContinuous(root_attributes, None)
    test_result = root_node.run_test(5.0)
    assert test_result == "A <= 10.0"

def test_dnc_over_threshold(root_attributes):
    """ running test with attribute below the threshold """
    root_node = DecisionNodeContinuous(root_attributes, None)
    test_result = root_node.run_test(15.0)
    assert test_result == "A > 10.0"

def test_dnc_on_threshold(root_attributes):
    """ running test with attribute below the threshold """
    root_node = DecisionNodeContinuous(root_attributes, None)
    test_result = root_node.run_test(10.0)
    assert test_result == "A <= 10.0"

def test_dnc_below_threshold_negative(root_attributes_negative):
    """ running test with attribute above the threshold and negative """
    root_node = DecisionNodeContinuous(root_attributes_negative, None)
    test_result = root_node.run_test(-5.0)
    assert test_result == "A > -10.0"

def test_dnc_over_threshold_negative(root_attributes_negative):
    """ running test with attribute below the threshold and negative """
    root_node = DecisionNodeContinuous(root_attributes_negative, None)
    test_result = root_node.run_test(-15.0)
    assert test_result == "A <= -10.0"

def test_dnc_on_threshold_negative(root_attributes_negative):
    """ running test with attribute on the threshold and negative """
    root_node = DecisionNodeContinuous(root_attributes_negative, None)
    test_result = root_node.run_test(-10.0)
    assert test_result == "A <= -10.0"

def test_get_child(root_attributes, node_a_attributes, node_b_attributes):
    """ must return the node corrsponding to the attribute value """
    root_node = DecisionNodeContinuous(root_attributes, None)
    node_a = DecisionNodeContinuous(node_a_attributes, root_node)
    node_b = DecisionNodeContinuous(node_b_attributes, root_node)
    root_node.add_child(node_a)
    root_node.add_child(node_b)
    assert root_node.get_child(5.0) == node_a

def test_does_not_get_child(root_attributes, node_a_attributes, node_b_attributes):
    node_b_attributes.node_name = "A <= 10.0"
    root_node = DecisionNodeContinuous(root_attributes, None)
    node_a = DecisionNodeContinuous(node_a_attributes, root_node)
    node_b = DecisionNodeContinuous(node_b_attributes, root_node)
    root_node.add_child(node_a)
    root_node.add_child(node_b)
    with pytest.raises(RuntimeError):
        root_node.get_child(15.0)
