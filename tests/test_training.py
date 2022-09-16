import pytest
import pandas as pd
from DecisionTree import DecisionTree
from traininghandler import TrainingHandler
from attributes import TrainingAttributes


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
def attributes_map():
    attr = {"Outlook": "categorical", "Humidity": "continuous",
            "Windy": "boolean", "Temperature": "continuous"}
    return attr

def test_continuous_split(paper_dataset, attributes_map):
    dataset = paper_dataset[["Temperature", "target"]]
    decision_tree = DecisionTree(attributes_map)
    training_attributes = TrainingAttributes()
    decision_handler = TrainingHandler(decision_tree,
            dataset, training_attributes)
    decision_handler.split_dataset()
    root_node = decision_tree.get_root_node()
    expected_leaves_labels = {"Temperature <= 70.0", "Temperature <= 72.0",
             "Temperature <= 75.0", "Temperature > 75.0"}
    assert root_node.get_attribute() == "Temperature"
    leaves_labels = {leaf.get_label() for leaf in decision_tree.get_leaves_nodes()}
    assert leaves_labels == expected_leaves_labels

def test_categorical_split(paper_dataset, attributes_map):
    dataset = paper_dataset[["Outlook", "target"]]
    decision_tree = DecisionTree(attributes_map)
    training_attributes = TrainingAttributes()
    decision_handler = TrainingHandler(decision_tree,
            dataset, training_attributes)
    decision_handler.split_dataset()
    root_node = decision_tree.get_root_node()
    expected_leaves_labels = {"Outlook = sunny",
            "Outlook = rain", "Outlook = overcast"}
    assert root_node.get_attribute() == "Outlook"
    leaves_labels = {leaf.get_label() for leaf in decision_tree.get_leaves_nodes()}
    assert leaves_labels == expected_leaves_labels
