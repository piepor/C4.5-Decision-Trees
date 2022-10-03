import pytest
import numpy as np
import pandas as pd
from sklearn import metrics
from c4dot5.DecisionTreeClassifier import DecisionTreeClassifier


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
def paper_dataset_unknown():
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

def test_classifier_paper_training(paper_dataset, paper_attributes_map):
    decision_tree = DecisionTreeClassifier(paper_attributes_map)
    decision_tree.fit(paper_dataset)
    root_node = decision_tree.get_root_node()
    expected_leaves_labels = {"Humidity <= 75.0", "Humidity > 75.0",
             "Windy = False", "Windy = True", "Outlook = overcast"}
    assert root_node.get_attribute() == "Outlook"
    leaves_labels = {leaf.get_label() for leaf in decision_tree.get_leaves_nodes()}
    assert leaves_labels == expected_leaves_labels

def test_sunny_weight(paper_dataset_unknown, paper_attributes_map):
    decision_tree = DecisionTreeClassifier(paper_attributes_map, max_depth=1)
    decision_tree.fit(paper_dataset_unknown)
    expected_distribution = {
            "Play": np.round(2+5/13, 4),
            "Don't Play": 3}
    sunny_leaf = decision_tree.get_leaf_node("Outlook = sunny")[0]
    sunny_leaf_distribution = sunny_leaf.get_classes()
    assert sunny_leaf_distribution == expected_distribution

def test_predict(paper_dataset, paper_attributes_map):
    decision_tree = DecisionTreeClassifier(paper_attributes_map)
    decision_tree.fit(paper_dataset)
    predctions = decision_tree.predict(paper_dataset)
    assert metrics.accuracy_score(paper_dataset['target'], predctions) == 1.0
