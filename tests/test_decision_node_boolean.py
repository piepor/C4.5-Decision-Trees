import pytest
from c4dot5.attributes import AttributeType, DecisionNodeAttributes, NodeType
from c4dot5.nodes import DecisionNodeCategorical

@pytest.fixture
def root_attributes():
    return DecisionNodeAttributes(0, "root", NodeType.DECISION_NODE_CATEGORICAL,
            "A", AttributeType.BOOLEAN, None)

@pytest.fixture
def node_a_attributes():
    return DecisionNodeAttributes(1, "A = True", NodeType.DECISION_NODE_CATEGORICAL,
            "B", AttributeType.BOOLEAN, None)

@pytest.fixture
def node_b_attributes():
    return DecisionNodeAttributes(1, "A = False", NodeType.DECISION_NODE_CATEGORICAL,
            "C", AttributeType.BOOLEAN, None)

def test_run_test(root_attributes):
    """ test the returning string """
    root_node = DecisionNodeCategorical(root_attributes, None)
    test_result = "A = False"
    assert test_result == root_node.run_test(False)

def test_get_child(root_attributes, node_a_attributes, node_b_attributes):
    """ must return the node corrsponding to the attribute value """
    root_node = DecisionNodeCategorical(root_attributes, None)
    node_a = DecisionNodeCategorical(node_a_attributes, root_node)
    node_b = DecisionNodeCategorical(node_b_attributes, root_node)
    root_node.add_child(node_a)
    root_node.add_child(node_b)
    assert root_node.get_child(False) == node_b

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
