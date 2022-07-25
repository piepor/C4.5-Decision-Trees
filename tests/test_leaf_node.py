import pytest
import numpy as np
from nodes import LeafNode, DecisionNodeContinuous
from attributes import LeafNodeAttributes, AttributeType, DecisionNodeAttributes, NodeType
from leaf_classes import LeafClasses

@pytest.fixture
def root_node():
    root_attributes = DecisionNodeAttributes(0, "root", "A",
            NodeType.DECISION_NODE_CONTINUOUS, AttributeType.CONTINUOUS, 10.0)
    return DecisionNodeContinuous(root_attributes, None)

@pytest.fixture
def leaf_attributes():
    classes = LeafClasses({"target_a": 20, "target_b": 3})
    return LeafNodeAttributes(1, "A <= 10.0", NodeType.DECISION_NODE_CONTINUOUS, classes)

def test_get_label(root_node, leaf_attributes):
    leaf_node = LeafNode(leaf_attributes, root_node)
    root_node.add_child(leaf_node)
    assert leaf_node.get_label() == "A <= 10.0"

def test_get_class_name(root_node, leaf_attributes):
    leaf_node = LeafNode(leaf_attributes, root_node)
    root_node.add_child(leaf_node)
    assert leaf_node.get_class_name() == "target_a"

def test_get_class_names(root_node, leaf_attributes):
    leaf_node = LeafNode(leaf_attributes, root_node)
    root_node.add_child(leaf_node)
    assert list(leaf_node.get_classes().keys()) == ["target_a", "target_b"]

def test_get_class_examples(root_node, leaf_attributes):
    leaf_node = LeafNode(leaf_attributes, root_node)
    root_node.add_child(leaf_node)
    assert leaf_node.get_classes()["target_a"] == 20

def test_get_class_distribution(root_node, leaf_attributes):
    leaf_node = LeafNode(leaf_attributes, root_node)
    root_node.add_child(leaf_node)
    expected_distribution = {
            "target_a": np.round(20/23, 4),
            "target_b": np.round(3/23, 4)}
    assert leaf_node.get_classes_distribution() == expected_distribution

def test_get_leaf_purity(root_node, leaf_attributes):
    leaf_node = LeafNode(leaf_attributes, root_node)
    root_node.add_child(leaf_node)
    assert leaf_node.get_leaf_purity() == np.round(20/23, 4)
