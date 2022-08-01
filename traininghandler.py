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
        self.attr_type_to_decision_node = {
                AttributeType.CONTINUOUS: NodeType.DECISION_NODE_CONTINUOUS,
                AttributeType.CATEGORICAL: NodeType.DECISION_NODE_CATEGORICAL,
                AttributeType.BOOLEAN: NodeType.DECISION_NODE_CATEGORICAL
                }

    def split_dataset(self, data_input: pd.DataFrame):
        """
        Recursively splits a dataset until some conditions are met.
        decision tree adds the nodes
        """
        # categorical and boolean arguments can be selected only one time in a "line of succession"
        action, split_attribute = check_split(data_input, self.training_attributes, self._split_fn)
        # if split attribute does not exist then is a leaf
        if action == Actions.ADD_LEAF:
            # TODO add exception handling
            raise Exception("something")
        else:
            name_attr = split_attribute.name_attr
            attr_type = self.decision_tree.get_attributes[name_attr]
            node_type = self.attr_type_to_decision_node[attr_type]
            root_node_attr = DecisionNodeAttributes(
                    0, "root", node_type, split_attribute.attr_name,
                    split_attribute.attr_type, split_attribute.local_threshold)
            root_node = self.decision_tree.create_node(root_node_attr, None)
            self.decision_tree.add_root_node(root_node)
            self.split_fn[attr_type](root_node, data_input, split_attribute)

    def split_continuous(self, parent_node: Node, data_in: pd.DataFrame, split_attribute: SplitAttributes):
        threshold = get_total_threshold(
                self.complete_dataset[split_attribute.name_attr], split_attribute.local_threshold)
        data_known = data_in[data_in[split_attribute] != '?']
        data_unknown = data_in[data_in[split_attribute] == '?']

        # lower than the threshold
        weight_unknown = len(data_known[data_known[split_attribute] <= threshold]) / len(data_known)
        new_weight = (np.array([weight_unknown] * len(data_unknown)) * np.array(data_unknown['weight'].copy(deep=True))).tolist()
        new_data_unknown = data_unknown.copy(deep=True)
        new_data_unknown.loc[:, ['weight']] = new_weight
        # concat the unknown data to the known ones, weighted, and pass to the next split
        data_low = pd.concat([data_known[data_known[split_attribute] <= threshold], new_data_unknown], ignore_index=True)
        # check the split to know what kind of node we have to add
        action, split_attribute_low = check_split(data_low, self.training_attributes, self._split_fn)
        if action == Actions.ADD_LEAF:
            # TODO add exception handling
            leaf_attr = LeafNodeAttributes(
                    parent_node.get_level(), parent_node.get_label(),
                    NodeType.LEAF_NODE, data_low.groupby("target")["weight"].sum()to_dict())
            node = self.decision_tree.create_node(leaf_attr, parent_node)
            self.decision_tree.add_node(node)
        else:
            node, attr_type = self.node_creation(parent_node, f"{parent_node.get_attribute()} <= {threshold}", split_attribute)
            self.split_fn[attr_type](node, data_low, split_attribute_low)

        # Higher than the threshold
        weight_unknown = len(data_known[data_known[split_attribute] > threshold]) / len(data_known)
        new_weight = (np.array([weight_unknown] * len(data_unknown)) * np.array(data_unknown['weight'].copy(deep=True))).tolist()
        new_data_unknown = data_unknown.copy(deep=True)
        new_data_unknown.loc[:, ['weight']] = new_weight
        data_high = pd.concat([data_known[data_known[split_attribute] > threshold], new_data_unknown], ignore_index=True)
        # check the split to know what kind of node we have to add
        action, split_attribute_high = check_split(data_high, self.training_attributes, self._split_fn)
        if action == Actions.ADD_LEAF:
            # TODO add exception handling
            leaf_attr = LeafNodeAttributes(
                    parent_node.get_level(), parent_node.get_label(),
                    NodeType.LEAF_NODE, data_high.groupby("target")["weight"].sum()to_dict())
            node = self.decision_tree.create_node(leaf_attr, parent_node)
            self.decision_tree.add_node(node)
        else:
            node, attr_type = self.node_creation(parent_node, f"{parent_node.get_attribute()} > {threshold}", split_attribute)
            self.split_fn[attr_type](node, data_low, split_attribute_low)

    def split_categorical(self, parent_node: Node, data_in: pd.DataFrame, split_attribute: SplitAttributes):
        data_known = data_in[data_in[split_attribute] != '?']
        data_unknown = data_in[data_in[split_attribute] == '?']
        for attr_value in data_known[split_attribute.attr_name].unique():
            # divide data
            weight_unknown = len(data_known[data_known[split_attribute] == attr_value]) / len(data_known)
            new_weight = (np.array([weight_unknown] * len(data_unknown)) * np.array(data_unknown['weight'].copy(deep=True))).tolist()
            new_data_unknown = data_unknown.copy(deep=True)
            new_data_unknown.loc[:, ['weight']] = new_weight
            # concat the unknown data to the known ones, weighted, and pass to the next split
            data = pd.concat([data_known[data_known[split_attribute] == attr_value], new_data_unknown], ignore_index=True)
            action, split_attribute_child = check_split(data, self.training_attributes, self._split_fn)
            if action == Actions.ADD_LEAF:
                # TODO add exception handling
                leaf_attr = LeafNodeAttributes(
                        parent_node.get_level(), parent_node.get_label(),
                        NodeType.LEAF_NODE, data.groupby("target")["weight"].sum()to_dict())
                node = self.decision_tree.create_node(leaf_attr, parent_node)
                self.decision_tree.add_node(node)
            else:
                node, attr_type = self.node_creation(parent_node, f"{parent_node.get_attribute()} = {split_attribute.attr_name}", split_attribute)
                self.split_fn[attr_type](node, data, split_attribute_child)

    def node_creation(self, parent_node: Node, node_name: str, split_attribute: SplitAttributes):
        """ create the node corresponding to split attribute and goes on splitting """
        name_attr = split_attribute.name_attr
        attr_type = self.decision_tree.get_attributes[name_attr]
        node_type = self.attr_type_to_decision_node[attr_type]
        node_attr = DecisionNodeAttributes(
                parent_node.get_level()+1, node_name, node_type, split_attribute.attr_name,
                split_attribute.attr_type, split_attribute.local_threshold)
        node = self.decision_tree.create_node(node_attr, parent_node)
        self.decision_tree.add_node(node)
        return node, attr_type
