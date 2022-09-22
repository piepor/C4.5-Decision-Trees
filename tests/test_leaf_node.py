import pytest
import numpy as np
from c4dot5.nodes import LeafNode, DecisionNodeContinuous
from c4dot5.attributes import LeafNodeAttributes, AttributeType
from c4dot5.attributes import DecisionNodeAttributes, NodeType
from c4dot5.node_utils import get_distribution

@pytest.fixture
def root_node():
    root_attributes = DecisionNodeAttributes(0, "root", "A",
            NodeType.DECISION_NODE_CONTINUOUS, AttributeType.CONTINUOUS, 10.0)
    return DecisionNodeContinuous(root_attributes, None)

@pytest.fixture
def leaf_attributes():
    classes = {"target_a": 20, "target_b": 3}
    return LeafNodeAttributes(1, "A <= 10.0", NodeType.DECISION_NODE_CONTINUOUS, classes)

def test_get_label(root_node, leaf_attributes):
    leaf_node = LeafNode(leaf_attributes, root_node)
    root_node.add_child(leaf_node)
    assert leaf_node.get_label() == "A <= 10.0"

def test_get_class_names(root_node, leaf_attributes):
    leaf_node = LeafNode(leaf_attributes, root_node)
    root_node.add_child(leaf_node)
    assert leaf_node.get_classes_names() == ["target_a", "target_b"]

def test_get_class_distribution(root_node, leaf_attributes):
    leaf_node = LeafNode(leaf_attributes, root_node)
    root_node.add_child(leaf_node)
    expected_distribution = {
            "target_a": np.round(20/23, 4),
            "target_b": np.round(3/23, 4)}
    assert get_distribution(leaf_node.get_classes()) == expected_distribution

def test_get_leaf_purity(root_node, leaf_attributes):
    leaf_node = LeafNode(leaf_attributes, root_node)
    root_node.add_child(leaf_node)
    assert leaf_node.get_purity() == np.round(20/23, 4)

def test_get_number_of_instances(root_node, leaf_attributes):
    leaf_node = LeafNode(leaf_attributes, root_node)
    root_node.add_child(leaf_node)
    assert leaf_node.get_instances_number() == 23
