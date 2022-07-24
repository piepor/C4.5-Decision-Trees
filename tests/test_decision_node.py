from nodes import DecisionNode
from attributes import NodeAttributes, AttributeType
from predicting import continuous_test_fn
import pytest

@pytest.fixture
def root_attributes():
    return NodeAttributes(0, "root", "A", AttributeType.CONTINUOUS, 10.0)

@pytest.fixture
def node_a_attributes():
    return NodeAttributes(1, "A <= 10", "B", AttributeType.CONTINUOUS, 10.0)

@pytest.fixture
def node_b_attributes():
    return NodeAttributes(1, "A > 10", "B", AttributeType.CONTINUOUS, 10.0)

def test_dn_negative_level(root_attributes):
    """ level cannot be negative """
    root_node = DecisionNode(root_attributes, None)
    root_attributes.node_level = -1
    with pytest.raises(ValueError):
        node = DecisionNode(root_attributes, root_node)

def test_root_level(root_attributes):
    """ root must have level 0 """
    root_attributes.node_level = 1
    with pytest.raises(ValueError):
        root_node = DecisionNode(root_attributes, None)

def test_type_continuous_has_threshold(root_attributes):
    """ a continuous type must have a threshold """
    root_attributes.threshold = None
    with pytest.raises(ValueError):
        root_node = DecisionNode(root_attributes, None)

def test_has_threshold_but_not_continuous(root_attributes):
    """ a continuous type must have a threshold """
    root_attributes.attribute_type = AttributeType.CATEGORICAL
    with pytest.raises(ValueError):
        root_node = DecisionNode(root_attributes, None)

def test_adding_child(root_attributes, node_a_attributes):
    """ add a child """
    root_node = DecisionNode(root_attributes, None)
    node_a = DecisionNode(node_a_attributes, root_node)
    root_node.add_child(node_a)
    assert root_node.get_childs() == set([node_a])

def test_adding_childs(root_attributes, node_a_attributes, node_b_attributes):
    """ add two childs """
    root_node = DecisionNode(root_attributes, None)
    node_a = DecisionNode(node_a_attributes, root_node)
    node_b = DecisionNode(node_b_attributes, root_node)
    root_node.add_child(node_a)
    root_node.add_child(node_b)
    assert root_node.get_childs() == set([node_a, node_b])

def test_delete_child(root_attributes, node_a_attributes, node_b_attributes):
    """ delete a child """
    root_node = DecisionNode(root_attributes, None)
    node_a = DecisionNode(node_b_attributes, root_node)
    node_b = DecisionNode(node_b_attributes, root_node)
    root_node.add_child(node_a)
    root_node.add_child(node_b)
    root_node.delete_child(node_a)
    assert root_node.get_childs() == set([node_b])

def test_add_itself(root_attributes):
    """ a node can't be the parent of itself """
    root_node = DecisionNode(root_attributes, None)
    with pytest.raises(ValueError):
        root_node.add_child(root_node)
