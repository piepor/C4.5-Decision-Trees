""" Functions related to the general training phase """

from enum import Enum, auto
import pandas as pd
import numpy as np
from c4dot5.nodes import DecisionNodeCategorical
from c4dot5.nodes import DecisionNodeContinuous, LeafNode, Node, DecisionNode
from c4dot5.attributes import NodeAttributes, SplitAttributes, DecisionNodeAttributes, LeafNodeAttributes


class Actions(Enum):
    """ Actions available to the decision tree """
    ADD_LEAF = auto()
    SPLIT_NODE = auto()

def create_continuous_decision_node(node_attributes: DecisionNodeAttributes, parent_node: Node) -> DecisionNode:
    """ create a continuous decision node """
    return DecisionNodeContinuous(node_attributes, parent_node)


def create_categorical_decision_node(node_attributes: DecisionNodeAttributes, parent_node: Node) -> DecisionNode:
    """ create a continuous decision node """
    return DecisionNodeCategorical(node_attributes, parent_node)


def create_leaf_node(node_attributes: LeafNodeAttributes, parent_node: Node) -> LeafNode:
    """ create a continuous decision node """
    return LeafNode(node_attributes, parent_node)

def class_entropy(data) -> float:
    """ Returns the weighted entropy of a split """
    ops = data.groupby('target')['weight'].sum() / data['weight'].sum()
    return - np.sum(ops * np.log2(ops))

def extract_max_gain_attributes(data: pd.DataFrame, split_attr: SplitAttributes) -> SplitAttributes:
    """ extract the attributes of the split with the max gain """
    max_idx = data['gain_ratio'].idxmax()
    split_attr.gain_ratio = data.iloc[max_idx]['gain_ratio']
    split_attr.info_gain = data.iloc[max_idx]['info_gain']
    split_attr.at_least_two = data.iloc[max_idx]['not_near_trivial_subset']
    split_attr.attr_name = data.iloc[max_idx]['attribute']
    split_attr.local_threshold = data.iloc[max_idx]['threshold']
    split_attr.threshold = data.iloc[max_idx]['threshold']
    # check if threshold is nan
    if split_attr.local_threshold:
        if np.isnan(split_attr.local_threshold):
            split_attr.local_threshold = None
            split_attr.threshold = None
    split_attr.errs_perc = data.iloc[max_idx]['errs_perc']
    return split_attr

def compute_local_threshold_gain(data_in: pd.DataFrame, threshold: float,
        attr_name: str, split_gain: float) -> tuple[float, float]:
    """ compute infomation gain and split infomation """
    freq_attr = data_in[data_in[attr_name] <= threshold][attr_name].count() / len(data_in)
    class_entropy_low = class_entropy(
            data_in[data_in[attr_name] <= threshold][['target', 'weight']])
    class_entropy_high = class_entropy(
            data_in[data_in[attr_name] > threshold][['target', 'weight']])
    split_gain_threshold = split_gain - freq_attr * class_entropy_low \
            - (1 - freq_attr) * class_entropy_high
    split_info = - freq_attr * np.log2(freq_attr) - (1 - freq_attr) * np.log2(1 - freq_attr)
    return split_gain_threshold, split_info

def are_there_at_least_two(len_subsets: list[int], min_instances: int):
    """ checks if in the subsets are present at least two subset with
    more samples than min_instances """
    return len([True for len_subset in len_subsets if len_subset >= min_instances]) >= 2

def get_total_threshold(data, local_threshold) -> float:
    """ Computes the threshold on the total dataset

    The global threshold is the maximum number less then or equal to the local one
    """
    data = data[data != '?'].astype(float)
    return data[data.le(local_threshold)].max()

def substitute_nan(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.fillna('?')
