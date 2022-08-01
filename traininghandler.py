import pandas as pd
from attributes import AttributeType, TrainingAttributes,
from attributes import DecisionNodeAttributes, LeafNodeAttributes
from nodes import Node
from decision_tree_refactor import DecisionTree
from training import check_split, Actions


class TrainingHandler:
    """ Class responsible to handle the training of the decision tree """
    def __init__(self,
            decision_tree: DecisionTree,
            complete_dataset: pd.DataFrame,
            training_attributes: TrainingAttributes):
        self.decision_tree = decision_tree
        self.complete_dataset = complete_dataset
        self.training_attributes = training_attributes
        self.split_fn = {
                AttributeType.CONTINUOUS: self.split_continuous,
                AttributeType.CATEGORICAL: self.split_categorical
                }

    def split_dataset(self, parent_node: Node, data_input: pd.DataFrame):
        """
        Recursively splits a dataset based until some conditions are met.
        decision tree adds the nodes
        """
        # categorical and boolean arguments can be selected only one time in a "line of succession"
        action, split_attribute = check_split(data_input, self.training_attributes, self._split_fn)
        # if split attribute does not exist then is a leaf
        if action == Actions.ADD_LEAF:
            # TODO add exception handling
            # TODO leaf attributes 
            leaf_attributes = LeafNodeAttributes(data_input["target"].value_counts())
            node = self.decision_tree.create_node(leaf_attributes)
            self.decision_tree.add_node(node)
        else:
            self.split_fn[self.decision_tree.attribute[split_attribute.name_attr](parent_node, data_input)]

    def split_continuous(self, parent_node: Node, data_in: pd.DataFrame):
        decision_node_attributes = DecisionNodeAttributes(attr_name, attr_type, split_attribute.local_threshold)
        node = self.decision_tree.create_node(decision_node_attributes)
        self.decision_tree.add_node(node)
        threshold = get_total_threshold(
                self.complete_dataset[split_attribute.name_attr], split_attribute.local_threshold)
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

    def split_categorical(self):
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
