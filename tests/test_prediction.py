import pytest
import pandas as pd
from sklearn import metrics
from c4dot5.DecisionTree import DecisionTree
from c4dot5.traininghandler import TrainingHandler
from c4dot5.attributes import TrainingAttributes
from c4dot5.predictor import PredictionHandler


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

def test_get_child_categorical(paper_dataset, paper_attributes_map):
    decision_tree = DecisionTree(paper_attributes_map)
    training_attributes = TrainingAttributes()
    training_handler = TrainingHandler(decision_tree,
            training_attributes)
    training_handler.split_dataset(paper_dataset)
    node = decision_tree.get_root_node()
    child = node.get_child('sunny')
    assert child.get_label() == 'Outlook = sunny'

def test_get_child_continuous(paper_dataset, paper_attributes_map):
    decision_tree = DecisionTree(paper_attributes_map)
    training_attributes = TrainingAttributes()
    training_handler = TrainingHandler(decision_tree,
            training_attributes)
    training_handler.split_dataset(paper_dataset)
    node = [node for node in decision_tree.get_nodes() if node.get_label() == 'Outlook = sunny'][0]
    child = node.get_child(70)
    assert child.get_label() == 'Humidity <= 75.0'

def test_predict(paper_dataset, paper_attributes_map):
    decision_tree = DecisionTree(paper_attributes_map)
    training_attributes = TrainingAttributes()
    training_handler = TrainingHandler(decision_tree,
            training_attributes)
    prediction_handler = PredictionHandler(decision_tree.get_leaves_nodes())
    training_handler.split_dataset(paper_dataset)
    predictions, _ = prediction_handler.predict(paper_dataset, decision_tree.get_root_node())
    assert metrics.accuracy_score(paper_dataset['target'], predictions) == 1.0
