import pandas as pd
import pytest
import numpy as np
from decision_tree_refactor import DecisionTree
from attributes import DecisionTreeAttributes, DecisionNodeAttributes
from attributes import NodeType, AttributeType, LeafNodeAttributes
from nodes import DecisionNodeContinuous
from predictor import PredictionHandler


@pytest.fixture
def decision_tree():
    attributes_map = {"attr1": "continuous"}
    decision_tree_attributes = DecisionTreeAttributes(attributes_map)
    return DecisionTree(decision_tree_attributes)


@pytest.fixture
def root_attributes():
    return DecisionNodeAttributes(0, "root", NodeType.DECISION_NODE_CONTINUOUS,
            "attr1", AttributeType.CONTINUOUS, 10.0)


@pytest.fixture
def node_a_attributes():
    return DecisionNodeAttributes(1, "attr1 <= 10.0", NodeType.DECISION_NODE_CONTINUOUS,
            "B", AttributeType.CONTINUOUS, 10.0)

@pytest.fixture
def node_b_attributes():
    return DecisionNodeAttributes(1, "attr1 > 10.0", NodeType.DECISION_NODE_CONTINUOUS,
            "C", AttributeType.CONTINUOUS, 10.0)

@pytest.fixture
def node_a_attributes_leave():
    return LeafNodeAttributes(1, "attr1 <= 10.0", NodeType.LEAF_NODE,
            {"target_a": 5, "target_b": 2})

@pytest.fixture
def node_b_attributes_leave():
    return LeafNodeAttributes(1, "attr1 > 10.0", NodeType.LEAF_NODE,
            {"target_a": 2, "target_b": 5})

@pytest.fixture
def paper_dataset():
    # df from the paper c4.5
    dataframe = pd.DataFrame(
            {'Outlook': ['sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'overcast',
                'overcast', 'overcast', 'overcast', 'rain', 'rain', 'rain', 'rain', 'rain'],
        'Temperature': [75, 80, 85, 72, 69, 72, 83, 64, 81, 71, 65, 75, 68, 70],
        'Humidity': [70, 90, 85, 95, 70, 90, 78, 65, 75, 80, 70, 80, 80, 96],
        'Windy': [True, True, False, False, False, True, False,
            True, False, True, True, False, False, False],
        'target': ["Play", "Don't Play", "Don't Play", "Don't Play", "Play", "Play", "Play",
            "Play", "Play", "Don't Play", "Don't Play", "Play", "Play", "Play"]})
    return dataframe

@pytest.fixture
def paper_tree():
    attributes_map = {
            "Outlook": "categorical", "Humidity": "continuous",
            "Windy": "boolean", "Temperature": "continuous"}
    attributes_tree = DecisionTreeAttributes(attributes_map)
    paper_tree = DecisionTree(attributes_tree)
    root_node_attr = DecisionNodeAttributes(0, "root", NodeType.DECISION_NODE_CATEGORICAL,
            "Outlook", AttributeType.CATEGORICAL)
    sunny_node_attr = DecisionNodeAttributes(1, "Outlook = sunny",
            NodeType.DECISION_NODE_CONTINUOUS,
            "Humidity", AttributeType.CONTINUOUS, float(75))
    humidity_low_node_attr = LeafNodeAttributes(2, f"Humidity <= {float(75)}", NodeType.LEAF_NODE,
            {"Play": 2})
    humidity_high_node_attr = LeafNodeAttributes(2, f"Humidity > {float(75)}", NodeType.LEAF_NODE,
            {"Don't Play": 3})
    overcast_node_attr = LeafNodeAttributes(1, "Outlook = overcast", NodeType.LEAF_NODE,
            {"Play": 4})
    rain_node_attr = DecisionNodeAttributes(1, "Outlook = rain", NodeType.DECISION_NODE_CATEGORICAL,
            "Windy", AttributeType.BOOLEAN)
    windy_true_node_attr = LeafNodeAttributes(2, "Windy = True", NodeType.LEAF_NODE,
            {"Don't Play": 2})
    windy_false_node_attr = LeafNodeAttributes(2, "Windy = False", NodeType.LEAF_NODE,
            {"Play": 3})
    root_node = paper_tree.create_node(root_node_attr, None)
    paper_tree.add_root_node(root_node)
    sunny_node = paper_tree.create_node(sunny_node_attr, root_node)
    paper_tree.add_node(sunny_node)
    humidity_low_node = paper_tree.create_node(humidity_low_node_attr, sunny_node)
    paper_tree.add_node(humidity_low_node)
    humidity_high_node = paper_tree.create_node(humidity_high_node_attr, sunny_node)
    paper_tree.add_node(humidity_high_node)
    overcast_node = paper_tree.create_node(overcast_node_attr, root_node)
    paper_tree.add_node(overcast_node)
    rain_node = paper_tree.create_node(rain_node_attr, root_node)
    paper_tree.add_node(rain_node)
    windy_true_node = paper_tree.create_node(windy_true_node_attr, rain_node)
    paper_tree.add_node(windy_true_node)
    windy_false_node = paper_tree.create_node(windy_false_node_attr, rain_node)
    paper_tree.add_node(windy_false_node)
    paper_tree.prediction_handler = PredictionHandler(paper_tree.get_leaves_nodes())
    return paper_tree

def test_add_node(decision_tree, root_attributes, node_a_attributes):
    root_node = decision_tree.create_node(root_attributes, None)
    decision_tree.add_root_node(root_node)
    node_a = decision_tree.create_node(node_a_attributes, root_node)
    decision_tree.add_node(node_a)
    assert root_node in decision_tree.get_nodes()
    assert node_a.get_parent_node() == root_node
    assert root_node.get_child(5.0) == node_a


def test_delete_node(decision_tree, root_attributes, node_a_attributes):
    root_node = decision_tree.create_node(root_attributes, None)
    decision_tree.add_root_node(root_node)
    node_a = decision_tree.create_node(node_a_attributes, root_node)
    decision_tree.add_node(node_a)
    decision_tree.delete_node(node_a)
    assert root_node in decision_tree.get_nodes()
    assert node_a not in decision_tree.get_nodes()
    assert node_a not in root_node.get_children()


def test_add_root_node(decision_tree, root_attributes, node_a_attributes):
    root_node = decision_tree.create_node(root_attributes, None)
    decision_tree.add_root_node(root_node)
    assert root_node == decision_tree.get_root_node()

def test_get_leaves_nodes(decision_tree, root_attributes, node_a_attributes_leave, node_b_attributes_leave):
    root_node = decision_tree.create_node(root_attributes, None)
    decision_tree.add_root_node(root_node)
    node_a = decision_tree.create_node(node_a_attributes_leave, root_node)
    decision_tree.add_node(node_a)
    node_b = decision_tree.create_node(node_b_attributes_leave, root_node)
    decision_tree.add_node(node_b)
    leaves = decision_tree.get_leaves_nodes()
    assert leaves == {node_a, node_b}

def test_predictions_distribution(decision_tree, root_attributes, node_a_attributes_leave, node_b_attributes_leave):
    root_node = decision_tree.create_node(root_attributes, None)
    decision_tree.add_root_node(root_node)
    node_a = decision_tree.create_node(node_a_attributes_leave, root_node)
    decision_tree.add_node(node_a)
    node_b = decision_tree.create_node(node_b_attributes_leave, root_node)
    decision_tree.add_node(node_b)
    decision_tree.prediction_handler = PredictionHandler(decision_tree.get_leaves_nodes())
    data = pd.DataFrame.from_dict({"attr1": [5.0], "target": ["target_a"]})
    expected_distr = [{"target_a": np.round(5/7, 4), "target_b": np.round(2/7, 4)}]
    _, distr = decision_tree.predict(data)
    assert distr == expected_distr

def test_predict_known(paper_tree, paper_dataset):
    targets = paper_dataset['target']
    predictions, _ = paper_tree.predict(paper_dataset.drop(columns=['target']))
    assert predictions == targets.tolist()
