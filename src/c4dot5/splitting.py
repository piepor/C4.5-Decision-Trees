""" Functions related to the splitting of the dataset """

import pandas as pd
import numpy as np
from typing import Union
from c4dot5.attributes import SplitAttributes, TrainingAttributes
from c4dot5.training import extract_max_gain_attributes, class_entropy, Actions
from c4dot5.training import are_there_at_least_two, compute_local_threshold_gain
from c4dot5.filtering import filter_dataset_cat, filter_dataset_high, filter_dataset_low


def get_split(
        data_in: pd.DataFrame, attr_fn_map: dict,
        min_instances: int, attr_map: dict) -> SplitAttributes:
    """ Compute the best split of the input data """
    chosen_split_attributes = SplitAttributes(None, None, False)
    # if there is only the target column or there aren't data the split doesn't exist
    if len(data_in['target'].unique()) > 1 and len(data_in) > 0:
        # in order the split to be chosen,
        # its information gain must be at least equal to the mean of all the tests considered
        tests_examined = {'gain_ratio': [], 'info_gain': [], 'threshold': [],
                'attribute': [], 'not_near_trivial_subset': [], 'errs_perc': []}
        for column in data_in.columns:
            # gain ratio and threshold (if exist) for every feature
            if not column in ['target', 'weight'] and len(data_in[column].unique()) > 1:
                attr_type = attr_map[column]
                data_column = data_in[[column, 'target', 'weight']]
                split_attributes = attr_fn_map[attr_type](data_column, min_instances)
                tests_examined['gain_ratio'].append(split_attributes.gain_ratio)
                tests_examined['info_gain'].append(split_attributes.info_gain)
                tests_examined['threshold'].append(split_attributes.local_threshold)
                tests_examined['attribute'].append(column)
                tests_examined['not_near_trivial_subset'].append(split_attributes.at_least_two)
                tests_examined['errs_perc'].append(split_attributes.errs_perc)
        # select the best split
        tests_examined = pd.DataFrame.from_dict(tests_examined)
        mean_info_gain = tests_examined['info_gain'].mean()
        # two conditions for the split to be chosen
        gain_ratio_gt_mean = tests_examined['info_gain'] >= mean_info_gain
        not_near_trivial_subset = tests_examined['not_near_trivial_subset']
        select_max_gain_ratio = tests_examined[
                (gain_ratio_gt_mean) & (not_near_trivial_subset)].reset_index(drop=True)
        if len(select_max_gain_ratio) != 0:
            chosen_split_attributes = extract_max_gain_attributes(
                    select_max_gain_ratio, chosen_split_attributes)
        elif len(tests_examined[tests_examined['not_near_trivial_subset']]) != 0:
            # Otherwise 'select_max_gain_ratio' computed before is empty
            select_max_gain_ratio = tests_examined[tests_examined['not_near_trivial_subset']]
            chosen_split_attributes = extract_max_gain_attributes(
                    select_max_gain_ratio, chosen_split_attributes)
    return chosen_split_attributes

def get_split_gain_categorical(data_in: pd.DataFrame, min_instances: int) -> SplitAttributes:
    """ Computes the information gain, the gain ratio, the local threshold
    and the meaningfulness of the split

    the infomation gain is computed on known data (i.e. not '?') considering their weight
    (reflecting the presence of unknonw data in previous splits).
    The gain ratio is computed considering one more class if unknown data are present.
    For the split to be meaningful, it has to have at least two subsplits
    with more than min_instances example each.
    """
    attr_name = [col for col in data_in.columns if col not in ['target', 'weight']][0]
    split_gain = class_entropy(data_in[data_in[attr_name] != '?'][['target', 'weight']])
    split_info = 0
    at_least_two = False
    # if categorical number of split = number of attributes
    data_counts = data_in[attr_name].value_counts()
    # deals with unknown data
    total_count = len(data_in)
    known_count = len(data_in[data_in[attr_name] != '?'])
    freq_known = known_count / total_count
    for attr_value in data_in[attr_name].unique():
        if not attr_value == '?':
            freq_attr = data_counts[attr_value] / known_count
            split_gain -= freq_attr * class_entropy(
                    data_in[data_in[attr_name] == attr_value][['target', 'weight']])
            split_info += - freq_attr * np.log2(freq_attr)
        else:
            # one more class for the unknown data
            split_info += -(1 - freq_known) * np.log2(1 - freq_known)
    gain_ratio = (freq_known * split_gain) / split_info
    # check also if at least two of the subset contain at least two cases,
    # to avoid near-trivial splits
    len_subsets = list(data_in[attr_name].value_counts())
    at_least_two = are_there_at_least_two(len_subsets, min_instances)
    # split_gain = info_gain
    split_attributes = SplitAttributes(
            np.round(gain_ratio, 4), np.round(split_gain, 4), at_least_two, attr_name)
    split_attributes.errs_perc = np.round(compute_split_error_cat(data_in), 4)
    return split_attributes

def get_split_gain_continuous(data_in: pd.DataFrame, min_instances: int) -> SplitAttributes:
    """ Computes the information gain, the gain ratio, the local threshold
    and the meaningfulness of the split

    the infomation gain is computed on known data (i.e. not '?') considerind their weight
    (reflecting the presence of unknonw data in previous splits).
    The gain ratio is computed considering one more class if unknown data are present.
    For the split to be meaningful, it has to have at least two subsplits
    with more than min_instances example each.
    """
    attr_name = [col for col in data_in.columns if col not in ['target', 'weight']][0]
    split_gain = class_entropy(data_in[data_in[attr_name] != '?'][['target', 'weight']])
    split_attributes = SplitAttributes(0, 0, False, None)
    split_info = 0
    # deals wìth unknown data
    freq_known = len(data_in[data_in[attr_name] != '?']) / len(data_in)
    data_in = data_in[data_in[attr_name] != '?']
    # sorted and compute thresolds
    data_in_sorted = data_in[attr_name].sort_values()
    thresholds = data_in_sorted.unique()[1:] - (np.diff(data_in_sorted.unique()) / 2)
    for threshold in thresholds:
        split_gain_threshold, split_info = compute_local_threshold_gain(
                data_in, threshold, attr_name, split_gain)
        # one more class for the unknown data
        if freq_known < 1.0:
            split_info += - (1 - freq_known) * np.log2(1 - freq_known)
        gain_ratio_temp = (freq_known * split_gain_threshold) / split_info
        len_subsets = [len(data_in[data_in[attr_name] <= threshold]),
                len(data_in[data_in[attr_name] > threshold])]
        at_least_two = are_there_at_least_two(len_subsets, min_instances)
        # save if better threshold
        if gain_ratio_temp > split_attributes.gain_ratio and at_least_two:
            split_attributes.gain_ratio = np.round(gain_ratio_temp, 4)
            split_attributes.info_gain = np.round(split_gain_threshold, 4)
            split_attributes.at_least_two = at_least_two
            split_attributes.local_threshold = threshold
            split_attributes.threshold = threshold
            split_attributes.attr_name = attr_name
            split_attributes.errs_perc = np.round(compute_split_error_cont(data_in, threshold), 4)
    return split_attributes

def check_split(data_in: pd.DataFrame,
        attributes: TrainingAttributes,
        attr_fn_map: dict,
        attr_map: dict) -> tuple[Actions, Union[SplitAttributes, None]]:
    """ check the split on a node and tells the action to take """
    if data_in.empty:
        raise Exception("you should not be here")
    split_attributes = get_split(data_in, attr_fn_map, attributes.min_instances, attr_map)
    node_purity = data_in["target"].value_counts().max() / len(data_in)
    if not split_attributes.attr_name or node_purity > attributes.node_purity:
        return Actions.ADD_LEAF, None
    node_errs_perc = data_in['target'].value_counts().sum() - data_in['target'].value_counts().max()
    node_errs_perc = np.round(node_errs_perc / len(data_in), 4)
    child_errs_perc = split_attributes.errs_perc
    if child_errs_perc >= node_errs_perc:
        return Actions.ADD_LEAF, None
    return Actions.SPLIT_NODE, split_attributes

def compute_split_error_cont(data_in: pd.DataFrame, threshold: float) -> float:
    """
    Computes the error made by the split of a continuous attribute if predicting
    the most frequent class for every child born after it. data_in contains only
    the attribute and the target columns.
    NEW The returned error is the minimum error between the children, otherwise """
    attr_name = [column for column in data_in.columns if column != 'target'][0]
    # if continuous type the split is binary given by th threshold
    split_left = filter_dataset_low(data_in, attr_name, threshold)
    # pandas function to count the occurnces of the different value of target
    values_count = split_left.groupby(['target'])["weight"].sum()
    # errors given by the difference between the sum of all occurrences and the most frequent
    errors_left = values_count.sum() - values_count.max()
    # compute perc
    errors_left_perc = errors_left / len(split_left)
    split_right = filter_dataset_high(data_in, attr_name, threshold)
    values_count = split_right.groupby(['target'])["weight"].sum()
    # errors given by the difference between the sum of all occurrences and the most frequent
    errors_right = values_count.sum() - values_count.max()
    # compute perc
    errors_right_perc = errors_right / len(split_right)
#    total_child_error = errors_left + errors_right
#    return total_child_error/len(data_in)
    #return min(errors_left_perc, errors_right_perc)
    return (errors_left + errors_right) / len(data_in)

def compute_split_error_cat(data_in: pd.DataFrame) -> float:
#def compute_split_error_cat(data_in: pd.DataFrame, threshold: float=None) -> int:
    """
    Computes the error made by the split if predicting
    the most frequent class for every child born after it.
    NEW The returned error is the minimum error between the children, otherwise """
    attr_name = [column for column in data_in.columns if column != 'target'][0]
    # if continuous type the split is binary given by th threshold
    #total_child_error = 0
    errors = []
    for attr_value in data_in[attr_name].unique():
        split = filter_dataset_cat(data_in, attr_name, attr_value)
        values_count = split.groupby(['target'])["weight"].sum()
        #total_child_error += values_count.sum() - values_count.max()
        errors.append((values_count.sum() - values_count.max()) / len(split))
    #return total_child_error/len(data_in)
    #return min(errors)
    return sum(errors) / len(data_in)
