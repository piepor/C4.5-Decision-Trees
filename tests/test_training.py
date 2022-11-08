import pytest
import pandas as pd
from c4dot5.DecisionTree import DecisionTree
from c4dot5.traininghandler import TrainingHandler
from c4dot5.attributes import TrainingAttributes
from c4dot5.exceptions import SplitError, WrongSplitEvaluationFunction


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
def paper_attributes_map():
    attr = {"Outlook": "categorical", "Humidity": "continuous",
            "Windy": "boolean", "Temperature": "continuous"}
    return attr

@pytest.fixture
def wrong_out_entropy():
    return ['One', 'list']

@pytest.fixture
def wrong_out_type_entropy():
    return 'wrong_type'

def test_continuous_split():
    attr_map = {"feat": "continuous"}
    dataset = pd.DataFrame({
        "feat": [20, 25, 19, 18, 17, 26, 30, 33, 47, 50],
        "target": ["target_1", "target_1", "target_1", "target_1", "target_1",
            "target_2", "target_2", "target_2", "target_2", "target_2"]
        })
    decision_tree = DecisionTree(attr_map)
    training_attributes = TrainingAttributes()
    training_handler = TrainingHandler(
            decision_tree, training_attributes)
    training_handler.split_dataset(dataset)
    root_node = decision_tree.get_root_node()
    expected_leaves_labels = {"feat <= 25.0", "feat > 25.0"}
    assert root_node.get_attribute() == "feat"
    leaves_labels = {leaf.get_label() for leaf in decision_tree.get_leaves_nodes()}
    assert leaves_labels == expected_leaves_labels

def test_categorical_split(paper_dataset, paper_attributes_map):
    dataset = paper_dataset[["Outlook", "target"]]
    decision_tree = DecisionTree(paper_attributes_map)
    training_attributes = TrainingAttributes()
    training_handler = TrainingHandler(decision_tree,
            training_attributes)
    training_handler.split_dataset(dataset)
    root_node = decision_tree.get_root_node()
    expected_leaves_labels = {"Outlook = sunny",
            "Outlook = rain", "Outlook = overcast"}
    assert root_node.get_attribute() == "Outlook"
    leaves_labels = {leaf.get_label() for leaf in decision_tree.get_leaves_nodes()}
    assert leaves_labels == expected_leaves_labels

def test_paper_training(paper_dataset, paper_attributes_map):
    decision_tree = DecisionTree(paper_attributes_map)
    training_attributes = TrainingAttributes()
    training_handler = TrainingHandler(decision_tree,
            training_attributes)
    training_handler.split_dataset(paper_dataset)
    root_node = decision_tree.get_root_node()
    expected_leaves_labels = {"Humidity <= 75.0", "Humidity > 75.0",
             "Windy = False", "Windy = True", "Outlook = overcast"}
    assert root_node.get_attribute() == "Outlook"
    leaves_labels = {leaf.get_label() for leaf in decision_tree.get_leaves_nodes()}
    assert leaves_labels == expected_leaves_labels

def test_stop_split_level(paper_dataset, paper_attributes_map):
    decision_tree = DecisionTree(paper_attributes_map)
    training_attributes = TrainingAttributes(max_depth=1)
    training_handler = TrainingHandler(decision_tree,
            training_attributes)
    training_handler.split_dataset(paper_dataset)
    expected_leaves_labels = {"Outlook = sunny", "Outlook = overcast", "Outlook = rain"}
    leaves_labels = {leaf.get_label() for leaf in decision_tree.get_leaves_nodes()}
    assert leaves_labels == expected_leaves_labels

def test_stop_split_min_instances(paper_dataset, paper_attributes_map):
    decision_tree = DecisionTree(paper_attributes_map)
    training_attributes = TrainingAttributes(min_instances=4)
    training_handler = TrainingHandler(decision_tree,
            training_attributes)
    training_handler.split_dataset(paper_dataset)
    expected_leaves_labels = {"Outlook = sunny", "Outlook = overcast", "Outlook = rain"}
    leaves_labels = {leaf.get_label() for leaf in decision_tree.get_leaves_nodes()}
    assert leaves_labels == expected_leaves_labels

def test_stop_split_purity(paper_dataset, paper_attributes_map):
    decision_tree = DecisionTree(paper_attributes_map)
    training_attributes = TrainingAttributes(node_purity=0.6)
    training_handler = TrainingHandler(decision_tree,
            training_attributes)
    with pytest.raises(SplitError):
        training_handler.split_dataset(paper_dataset)

def test_wrong_eval_split_fn_cat(paper_dataset, paper_attributes_map, 
                                 wrong_out_entropy, wrong_out_type_entropy):
    dataset = paper_dataset[["Outlook", "target"]]
    decision_tree = DecisionTree(paper_attributes_map)
    training_attributes = TrainingAttributes()
    training_handler = TrainingHandler(decision_tree,
            training_attributes, evaluate_split_fn=wrong_out_entropy)
    with pytest.raises(WrongSplitEvaluationFunction):
        training_handler.split_dataset(dataset)
    training_handler = TrainingHandler(decision_tree,
            training_attributes, evaluate_split_fn=wrong_out_type_entropy)
    with pytest.raises(WrongSplitEvaluationFunction):
        training_handler.split_dataset(dataset)

def test_wrong_eval_split_fn_cont(wrong_out_entropy, wrong_out_type_entropy):
    attr_map = {"feat": "continuous"}
    dataset = pd.DataFrame({
        "feat": [20, 25, 19, 18, 17, 26, 30, 33, 47, 50],
        "target": ["target_1", "target_1", "target_1", "target_1", "target_1",
            "target_2", "target_2", "target_2", "target_2", "target_2"]
        })
    decision_tree = DecisionTree(attr_map)
    training_attributes = TrainingAttributes()
    training_handler = TrainingHandler(
            decision_tree, training_attributes, evaluate_split_fn=wrong_out_entropy)
    with pytest.raises(WrongSplitEvaluationFunction):
        training_handler.split_dataset(dataset)
    training_handler = TrainingHandler(
            decision_tree, training_attributes, evaluate_split_fn=wrong_out_type_entropy)
    with pytest.raises(WrongSplitEvaluationFunction):
        training_handler.split_dataset(dataset)
