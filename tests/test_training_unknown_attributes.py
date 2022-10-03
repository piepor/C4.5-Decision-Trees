import pytest
import pandas as pd
import numpy as np
from c4dot5.DecisionTree import DecisionTree
from c4dot5.traininghandler import TrainingHandler
from c4dot5.attributes import TrainingAttributes


@pytest.fixture
def paper_dataset():
    # df from the paper c4.5
    dataframe = pd.DataFrame(
            {'Outlook': ['sunny', 'sunny', 'sunny', 'sunny', 'sunny', '?',
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

def test_sunny_weight(paper_dataset, paper_attributes_map):
    decision_tree = DecisionTree(paper_attributes_map)
    training_attributes = TrainingAttributes(max_depth=1)
    training_handler = TrainingHandler(decision_tree,
            training_attributes)
    training_handler.split_dataset(paper_dataset)
    expected_distribution = {
            "Play": np.round(2+5/13, 4),
            "Don't Play": 3}
    sunny_leaf = decision_tree.get_leaf_node("Outlook = sunny")[0]
    sunny_leaf_distribution = sunny_leaf.get_classes()
    assert sunny_leaf_distribution == expected_distribution

def test_nan(paper_dataset, paper_attributes_map):
    decision_tree = DecisionTree(paper_attributes_map)
    training_attributes = TrainingAttributes(max_depth=1)
    training_handler = TrainingHandler(decision_tree,
            training_attributes)
    paper_dataset.at[5, 'Outlook'] = np.nan
    training_handler.split_dataset(paper_dataset)
    expected_distribution = {
            "Play": np.round(2+5/13, 4),
            "Don't Play": 3}
    sunny_leaf = decision_tree.get_leaf_node("Outlook = sunny")[0]
    sunny_leaf_distribution = sunny_leaf.get_classes()
    assert sunny_leaf_distribution == expected_distribution
