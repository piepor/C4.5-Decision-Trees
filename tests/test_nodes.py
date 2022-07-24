from nodes import DecisionNode
from attributes import NodeAttributes, AttributeType
from predicting import continuous_test_fn
import pytest

def test_dn_negative_level():
    """ level cannot be negative """
    root_attributes = NodeAttributes(0, "root", "A",
            AttributeType.CONTINUOUS, 10.0, continuous_test_fn)
    root_node = DecisionNode(root_attributes, None)
    node_attributes = NodeAttributes(-1, "A < 10", "B",
            AttributeType.CONTINUOUS, 10.0, continuous_test_fn)
    with pytest.raises(ValueError):
        node = DecisionNode(node_attributes, root_node)

def test_root_level():
    """ root must have level 0 """
    root_attributes = NodeAttributes(-1, "root", "A",
            AttributeType.CONTINUOUS, 10.0, continuous_test_fn)
    with pytest.raises(ValueError):
        root_node = DecisionNode(root_attributes, None)

def test_type_continuous_has_threshold():
    """ a continuous type must have a threshold """
    root_attributes = NodeAttributes(0, "root", "A",
            AttributeType.CONTINUOUS, None, continuous_test_fn)
    with pytest.raises(ValueError):
        root_node = DecisionNode(root_attributes, None)

def test_has_threshold_but_not_continuous():
    """ a continuous type must have a threshold """
    root_attributes = NodeAttributes(0, "root", "A",
            AttributeType.CATEGORICAL, 10, continuous_test_fn)
    with pytest.raises(ValueError):
        root_node = DecisionNode(root_attributes, None)

def test_adding_child():
    """ add a child """
    root_attributes = NodeAttributes(0, "root", "A",
            AttributeType.CONTINUOUS, 10, continuous_test_fn)
    root_node = DecisionNode(root_attributes, None)
    node_attributes_a = NodeAttributes(1, "A <= 10", "B",
            AttributeType.CONTINUOUS, 10.0, continuous_test_fn)
    node_a = DecisionNode(node_attributes_a, root_node)
    root_node.add_child(node_a)
    assert root_node.get_childs() == set([node_a])

def test_adding_childs():
    """ add two childs """
    root_attributes = NodeAttributes(0, "root", "A",
            AttributeType.CONTINUOUS, 10, continuous_test_fn)
    root_node = DecisionNode(root_attributes, None)
    node_attributes_a = NodeAttributes(1, "A <= 10", "B",
            AttributeType.CONTINUOUS, 10.0, continuous_test_fn)
    node_a = DecisionNode(node_attributes_a, root_node)
    node_attributes_b = NodeAttributes(1, "A > 10", "B",
            AttributeType.CONTINUOUS, 10.0, continuous_test_fn)
    node_b = DecisionNode(node_attributes_b, root_node)
    root_node.add_child(node_a)
    root_node.add_child(node_b)
    assert root_node.get_childs() == set([node_a, node_b])

def test_delete_child():
    """ delete a child """
    root_attributes = NodeAttributes(0, "root", "A",
            AttributeType.CONTINUOUS, 10, continuous_test_fn)
    root_node = DecisionNode(root_attributes, None)
    node_attributes_a = NodeAttributes(1, "A <= 10", "B",
            AttributeType.CONTINUOUS, 10.0, continuous_test_fn)
    node_a = DecisionNode(node_attributes_a, root_node)
    node_attributes_b = NodeAttributes(1, "A > 10", "B",
            AttributeType.CONTINUOUS, 10.0, continuous_test_fn)
    node_b = DecisionNode(node_attributes_b, root_node)
    root_node.add_child(node_a)
    root_node.add_child(node_b)
    root_node.delete_child(node_a)
    assert root_node.get_childs() == set([node_b])
