from decision_tree_refactor import DecisionTree
import pytest


def test_init():
    with pytest.raises(TypeError):
        DecisionTree({'attr1': 'float'})
