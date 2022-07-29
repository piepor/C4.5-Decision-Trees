""" functions for the prediction phase """
import numpy as np
from nodes import Node

def select_childs_for_prediction(attr_value: str | bool | float, node: Node) -> list[Node]:
    """ returns childs based on attribute known or unknown """
    if attr_value == "?":
        return node.get_childs()
    return [node.get_child(attr_value)]

def create_predictions_dict(leaves_nodes: list[Node]) -> dict:
    """ creates the dictionary of classes distribution """
    classes = []
    for leave in leaves_nodes:
        classes.extend(list(leave.get_classes().keys()))
    predictions_dict = dict.fromkeys(set(classes), [])
    return predictions_dict

def get_predictions_distribution(predictions_dict: dict) -> dict:
    """ Returns a dictionary containing the distribution over the target classes """
    distribution_dict = dict.fromkeys(predictions_dict.keys())
    total_count = 0
    for item in predictions_dict.items():
        total_count += sum(item[1])
    for item in predictions_dict.items():
        distribution_dict[item[0]] = np.round(sum(item[1]) / total_count, 4)
    return distribution_dict
