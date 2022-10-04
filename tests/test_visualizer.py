import os
from pathlib import Path
import pytest
import graphviz
import numpy as np
import pandas as pd
from sklearn import metrics
from c4dot5.DecisionTreeClassifier import DecisionTreeClassifier
from c4dot5.visualizer import Visualizer


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
def graph_paper():
    dot = graphviz.Digraph(name="Paper-tree", comment="Paper-tree")
    dot.node("root", "root \n [split attribute: Outlook]")
    dot.node("Outlook = overcast", "Outlook = overcast \n [Classes]: \n - Play: 4.0")
    dot.edge("root", "Outlook = overcast")
    dot.node("Outlook = rain", "Outlook = rain \n [split attribute: Windy]")
    dot.edge("root", "Outlook = rain")
    dot.node("Windy = False", "Windy = False \n [Classes]: \n - Play: 3.0")
    dot.edge("Outlook = rain", "Windy = False")
    dot.node("Windy = True", "Windy = True \n [Classes]: \n - Don't Play: 2.0")
    dot.edge("Outlook = rain", "Windy = True")
    dot.node("Outlook = sunny", "Outlook = sunny \n [split attribute: Humidity]")
    dot.edge("root", "Outlook = sunny")
    dot.node("Humidity <= 75.0", "Humidity <= 75.0 \n [Classes]: \n - Play: 2.0")
    dot.edge("Outlook = sunny", "Humidity <= 75.0")
    dot.node("Humidity > 75.0", "Humidity > 75.0 \n [Classes]: \n - Don't Play: 3.0")
    dot.edge("Outlook = sunny", "Humidity > 75.0")
    return dot

def test_visualizer(paper_dataset, paper_attributes_map, graph_paper):
    decision_tree = DecisionTreeClassifier(paper_attributes_map)
    decision_tree.fit(paper_dataset)
    visualizer = Visualizer(decision_tree, 'Paper-tree')
    visualizer.create_digraph()
    assert visualizer.dot.source == graph_paper.source

def test_save_view(paper_dataset, paper_attributes_map, graph_paper):
    decision_tree = DecisionTreeClassifier(paper_attributes_map)
    decision_tree.fit(paper_dataset)
    visualizer = Visualizer(decision_tree, 'Paper-tree')
    visualizer.create_digraph()
    visualizer.save_view('temp', view=False)
    # image
    assert Path(os.path.join(os.getcwd(), 'temp', 'Paper-tree.gv.png')).exists()
    # source
    assert Path(os.path.join(os.getcwd(), 'temp', 'Paper-tree.gv')).exists()
    Path(os.path.join(os.getcwd(), 'temp', 'Paper-tree.gv.png')).unlink()
    Path(os.path.join(os.getcwd(), 'temp', 'Paper-tree.gv')).unlink()
    Path(os.path.join(os.getcwd(), 'temp')).rmdir()
