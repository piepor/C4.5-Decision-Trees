""" functions for the prediction phase """
from copy import deepcopy
import numpy as np
from c4dot5.nodes import Node

def select_children_for_prediction(attr_value: str | bool | float, node: Node) -> list[Node]:
    """ returns childs based on attribute known or unknown """
    if attr_value == "?":
        return node.get_children()
    return [node.get_child(attr_value)]

def create_predictions_dict(leaves_nodes: list[Node]) -> dict:
    """ creates the dictionary of classes distribution """
    leaves_names = []
    for leave in leaves_nodes:
        leaves_names.append(leave.get_label())
    predictions_dict = dict.fromkeys(leaves_names, [])
    return predictions_dict

def create_distribution_dict(pred_dict: dict) -> dict:
    """ returns a dictionary with the targets involved """
    distribution_targets = set()
    for leaf_node in pred_dict:
        if pred_dict[leaf_node]:
            for target in pred_dict[leaf_node]:
                distribution_targets.add(target)
    return dict.fromkeys(distribution_targets, 0)

def get_predictions_distribution(pred_dict: dict) -> dict:
    """ Returns a dictionary containing the distribution over the target classes """
    pred_dict = deepcopy(pred_dict)
    pred_dict = {key: pred_dict[key] for key in pred_dict if len(pred_dict[key]) > 0}
    distribution_dict = create_distribution_dict(pred_dict)
    total_count = 0
    for item in pred_dict.items():
        total_count += sum(item[1].values())
    # in the paper is as follows:
    # target_x: sum_leafcontainingx( samples_leaf / total_samples * perc_samples_in_leaf)
    for item in pred_dict.items():
        for target in item[1]:
            distribution_dict[target] += item[1][target] / total_count
    # round the result
    return {item[0]: np.round(item[1], 4) for item in distribution_dict.items()}
