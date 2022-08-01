from enum import Enum, auto
import pandas as pd
import numpy as np
from nodes import DecisionNodeCategorical
from nodes import DecisionNodeContinuous, LeafNode, Node
from attributes import NodeAttributes, SplitAttributes
from attributes import NodeType, TrainingAttributes


class Actions(Enum):
    """ Actions available to the decision tree """
    ADD_LEAF: auto()
    SPLIT_NODE: auto()

def create_continuous_decision_node(node_attributes: NodeAttributes, parent_node: Node) -> None:
    """ create a continuous decision node """
    return DecisionNodeContinuous(node_attributes, parent_node)


def create_categorical_decision_node(node_attributes: NodeAttributes, parent_node: Node) -> None:
    """ create a continuous decision node """
    return DecisionNodeCategorical(node_attributes, parent_node)


def create_leaf_node(node_attributes: NodeAttributes, parent_node: Node) -> None:
    """ create a continuous decision node """
    return LeafNode(node_attributes, parent_node)

def class_entropy(data) -> float:
    """ Returns the weighted entropy of a split """
    ops = data.groupby('target')['weight'].sum() / data['weight'].sum()
    return - np.sum(ops * np.log2(ops))

def get_split(data_in: pd.DataFrame, attr_fn_map: dict, min_instances: int) -> SplitAttributes:
    """ Compute the best split of the input data """
    chosen_split_attributes = SplitAttributes(None, None, False)
    # if there is only the target column or there aren't data the split doesn't exist
    if len(data_in['target'].unique()) > 1 and len(data_in) > 0:
        # in order the split to be chosen,
        # its information gain must be at least equal to the mean of all the tests considered
        tests_examined = {'gain_ratio': [], 'info_gain': [], 'threshold': [],
                'attribute': [], 'not_near_trivial_subset': []}
        for column in data_in.columns:
            # gain ratio and threshold (if exist) for every feature
            if not column in ['target', 'weight'] and len(data_in[column].unique()) > 1:
                data_column = data_in[[column, 'target', 'weight']]
                split_attributes = attr_fn_map[column](data_column, min_instances)
                tests_examined['gain_ratio'].append(split_attributes.gain_ratio)
                tests_examined['info_gain'].append(split_attributes.info_gain)
                tests_examined['threshold'].append(split_attributes.local_threshold)
                tests_examined['attribute'].append(column)
                tests_examined['not_near_trivial_subset'].append(split_attributes.at_least_two)
        # select the best split
        tests_examined = pd.DataFrame.from_dict(tests_examined)
        mean_info_gain = tests_examined['info_gain'].mean()
        # two conditions for the split to be chosen
        gain_ratio_gt_mean = tests_examined['info_gain'] >= mean_info_gain
        not_near_trivial_subset = tests_examined['not_near_trivial_subset']
        select_max_gain_ratio = tests_examined[
                (gain_ratio_gt_mean) & (not_near_trivial_subset)]
        if len(select_max_gain_ratio) != 0:
            chosen_split_attributes = extract_max_gain_attributes(
                    select_max_gain_ratio, chosen_split_attributes)
        elif len(tests_examined[tests_examined['not_near_trivial_subset']]) != 0:
            # Otherwise 'select_max_gain_ratio' computed before is empty
            select_max_gain_ratio = tests_examined[tests_examined['not_near_trivial_subset']]
            chosen_split_attributes = extract_max_gain_attributes(
                    select_max_gain_ratio, chosen_split_attributes)
    return chosen_split_attributes

def extract_max_gain_attributes(data: pd.DataFrame, split_attr: SplitAttributes) -> SplitAttributes:
    """ extract the attributes of the split with the max gain """
    max_idx = data['gain_ratio'].idxmax()
    split_attr.gain_ratio = data.loc[max_idx, 'gain_ratio']
    split_attr.threshold = data.loc[max_idx, 'threshold']
    split_attr.attr_name = data.loc[max_idx, 'attribute']
    return split_attr

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
    split_attributes = SplitAttributes(gain_ratio, split_gain, at_least_two)
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
            split_attributes.gain_ratio = gain_ratio_temp
            split_attributes.info_gain = split_gain_threshold
            split_attributes.at_least_two = at_least_two
            split_attributes.local_threshold = threshold
    return split_attributes

def compute_local_threshold_gain(data_in: pd.DataFrame, threshold: float,
        attr_name: str, split_gain: float) -> [float, float]:
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

def check_split(data_in: pd.DataFrame,
        attributes: TrainingAttributes,
        attr_fn_map: dict) -> [Actions, SplitAttributes]:
    """ check the split on a node and tells the action to take """
    split_attributes = get_split(data_in, attr_fn_map, attributes.min_instances)
    node_purity = data_in["target"].value_counts.max() / len(data_in)
    if not split_attributes.attr_name or node_purity > attributes.node_purity:
        return Actions.ADD_LEAF, None
    node_errs_perc = data_in['target'].value_counts().sum() - data_in['target'].value_counts().max()
    node_errs_perc = node_errs_perc / len(data_in)
    split_attributes = get_split(data_in, attr_fn_map, attributes.min_instances)
    child_errs_perc = compute_split_error(
            data_in[[split_attributes.attr_name, 'target']], split_attributes.local_threshold)
    if child_errs_perc >= node_errs_perc:
        return Actions.ADD_LEAF, None
    return Actions.SPLIT_NODE, split_attributes

def split_node( node: Node, data_in: pd.DataFrame, data_total) -> None:
    """
    Recurseviley split a node based on "data_in" until some conditions are met
    and a leaves nodes are added to the tree """
    # categorical and boolean arguments can be selected only one time in a "line of succession"
    node_type = node.get_node_type()
    if node_type in [NodeType.CATEGORICAL, NodeType.BOOLEAN] and node.get_label() == 'root':
        data_in = data_in.copy(deep=True)
        data_in = data_in.drop(columns=[node.get_label().split()[0]])
    max_gain_ratio, local_threshold, split_attribute = self.get_split(data_in)
    #breakpoint()
    # compute error predicting the most frequent class without splitting
    node_errors = data_in['target'].value_counts().sum() - data_in['target'].value_counts().max()
    # compute percentage
    # TODO directly compute percentage
    node_errors = node_errors / len(data_in)
    # if split attribute does not exist then is a leaf
    if split_attribute is not None and node.get_level() < self.max_depth:
        child_errors = self.compute_split_error(data_in[[split_attribute, 'target']], local_threshold)
        # compute percentage
        # TODO directly compute percentage
        child_errors = child_errors / len(data_in)
        # if child errors are greater the actual error of the node than the split is useless
        if child_errors >= node_errors:
            # the node (default type "DecisionNode") is "transformed" in a leaf node ("LeafNode" type)
            parent_node = node.get_parent_node()
            if parent_node is None and node.get_label() == 'root':
                print("Childs error percentage is higher than the root one. Can't find a suitable split of the root node.")
            elif parent_node is None:
                raise Exception("Can't transform DecisionNode {} in LeafNode: no parent found".format(node.get_label()))
            else:
            #breakpoint()
                self.delete_node(node)
                node = LeafNode(dict(data_in.groupby('target')['weight'].sum().round(4)), node.get_label(), node.get_level())
                self.add_node(node, parent_node)
        else:
            # if the attribute with the greatest gain is continuous than the split is binary
            if self._attributes_map[split_attribute] == 'continuous':
                # compute global threshold (on the complete dataset) from local one
                threshold = get_total_threshold(data_total[split_attribute], local_threshold)
                node.set_attribute('{}:{}'.format(split_attribute, threshold), 'continuous')
                # create DecisionNode, recursion and add node
                # Low split
                low_split_node = DecisionNode('{} <= {}'.format(split_attribute, float(threshold)), node.get_level())
                self.add_node(low_split_node, node)
                # the split is computed on the known data and then weighted on unknown ones
                data_known = data_in[data_in[split_attribute] != '?']
                data_unknown = data_in[data_in[split_attribute] == '?']
                weight_unknown = len(data_known[data_known[split_attribute] <= threshold]) / len(data_known)
                new_weight = (np.array([weight_unknown] * len(data_unknown)) * np.array(data_unknown['weight'].copy(deep=True))).tolist()
                new_data_unknown = data_unknown.copy(deep=True)
                new_data_unknown.loc[:, ['weight']] = new_weight
                # concat the unknown data to the known ones, weighted, and pass to the next split
                new_data_low = pd.concat([data_known[data_known[split_attribute] <= threshold], new_data_unknown], ignore_index=True)
                self.split_node(low_split_node, new_data_low, data_total)
                # High split
                high_split_node = DecisionNode('{} > {}'.format(split_attribute, float(threshold)), node.get_level())
                self.add_node(high_split_node, node)
                weight_unknown = len(data_known[data_known[split_attribute] > threshold]) / len(data_known)
                new_weight = (np.array([weight_unknown] * len(data_unknown)) * np.array(data_unknown['weight'].copy(deep=True))).tolist()
                new_data_unknown = data_unknown.copy(deep=True)
                new_data_unknown.loc[:, ['weight']] = new_weight
                new_data_high = pd.concat([data_known[data_known[split_attribute] > threshold], new_data_unknown], ignore_index=True)
                self.split_node(high_split_node, new_data_high, data_total)
            else:
                # if the attribute is categorical or boolean than there is a node for every possible attribute value
                node.set_attribute(split_attribute, self._attributes_map[split_attribute])
                data_known = data_in[data_in[split_attribute] != '?']
                data_unknown = data_in[data_in[split_attribute] == '?']
                for attr_value in data_known[split_attribute].unique():
                    # create DecisionNode, recursion and add node
                    child_node = DecisionNode('{} = {}'.format(split_attribute, attr_value), node.get_level())
                    self.add_node(child_node, node)
                    # the split is computed on the known data and then weighted on unknown ones
                    weight_unknown = len(data_known[data_known[split_attribute] == attr_value]) / len(data_known)
                    new_weight = (np.array([weight_unknown] * len(data_unknown)) * np.array(data_unknown['weight'].copy(deep=True))).tolist()
                    new_data_unknown = data_unknown.copy(deep=True)
                    new_data_unknown.loc[:, ['weight']] = new_weight
                    # concat the unknown data to the known ones, weighted, and pass to the next split
                    new_data = pd.concat([data_known[data_known[split_attribute] == attr_value], new_data_unknown], ignore_index=True)
                    self.split_node(child_node, new_data, data_total)
    else:
        # the node (default type "DecisionNode") is "transformed" in a leaf node ("LeafNode" type)
        parent_node = node.get_parent_node()
        if parent_node is None and node.get_label() == 'root':
            print("The data are not feasible for fitting a tree. Can't find a suitable split of the root node.")
        elif parent_node is None:
            raise Exception("Can't transform DecisionNode {} in LeafNode: no parent found".format(node.get_label()))
        else:
            self.delete_node(node)
            # the final number of class contained is the sum of the weights of every row with that specific target
            node = LeafNode(dict(data_in.groupby('target')['weight'].sum().round(4)), node.get_label(), node.get_level())
            self.add_node(node, parent_node)

def compute_split_error(data_in, threshold) -> int:
    """ Computes the error made by the split if predicting the most frequent class for every child born after it """
    # compute percentage
    # TODO directly compute percentage
    attr_name = [column for column in data_in.columns if column != 'target'][0]
    attr_type = self._attributes_map[attr_name]
    # if continuous type the split is binary given by th threshold
    if attr_type == 'continuous':
        data_in_unknown = data_in[data_in[attr_name] != '?'].copy()
        data_in_unknown.loc[:, attr_name] = data_in_unknown[attr_name].astype(float).copy()
        #breakpoint()
        split_left = data_in_unknown[data_in_unknown[attr_name] <= threshold].copy()
        # pandas function to count the occurnces of the different value of target
        values_count = split_left['target'].value_counts()
        # errors given by the difference between the sum of all occurrences and the most frequent
        errors_left = values_count.sum() - values_count.max()
        split_right = data_in_unknown[data_in_unknown[attr_name] > threshold].copy()
        values_count = split_right['target'].value_counts()
        errors_right = values_count.sum() - values_count.max()
        total_child_error = errors_left + errors_right
    # if categorical or boolean, there is a child for every possible attribute value
    else:
        total_child_error = 0
        for attr_value in data_in[attr_name].unique():
            split = data_in[data_in[attr_name] == attr_value].copy()
            values_count = split['target'].value_counts()
            total_child_error += values_count.sum() - values_count.max()
    return total_child_error
