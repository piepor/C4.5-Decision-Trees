import pandas as pd
from c4dot5.attributes import AttributeType, TrainingAttributes, NodeType
from c4dot5.attributes import DecisionNodeAttributes, LeafNodeAttributes, SplitAttributes
from c4dot5.nodes import Node
from c4dot5.DecisionTree import DecisionTree
from c4dot5.training import Actions, get_total_threshold, substitute_nan
from c4dot5.filtering import filter_dataset_cat, filter_dataset_high, filter_dataset_low
from c4dot5.splitting import check_split, get_split_gain_categorical, get_split_gain_continuous
from c4dot5.exceptions import SplitError


class TrainingHandler:
    """ Class responsible to handle the training of the decision tree """
    def __init__(self,
            decision_tree: DecisionTree,
            #complete_dataset: pd.DataFrame,
            training_attributes: TrainingAttributes):
        self.decision_tree = decision_tree
        self.complete_dataset = None
        self.training_attributes = training_attributes
        self.get_split_fn = {
                AttributeType.CONTINUOUS: get_split_gain_continuous,
                AttributeType.CATEGORICAL: get_split_gain_categorical,
                AttributeType.BOOLEAN: get_split_gain_categorical
                }
        self.split_fn = {
                AttributeType.CONTINUOUS: self.split_continuous,
                AttributeType.CATEGORICAL: self.split_categorical,
                AttributeType.BOOLEAN: self.split_categorical
                }
        self.attr_type_to_decision_node = {
                AttributeType.CONTINUOUS: NodeType.DECISION_NODE_CONTINUOUS,
                AttributeType.CATEGORICAL: NodeType.DECISION_NODE_CATEGORICAL,
                AttributeType.BOOLEAN: NodeType.DECISION_NODE_CATEGORICAL
                }

    def split_dataset(self, dataset: pd.DataFrame):
        """
        Recursively splits a dataset until some conditions are met.
        decision tree adds the nodes
        """
        # substitute nan in dataframe with '?'
        dataset = substitute_nan(dataset.copy(deep=True))
        self.complete_dataset = dataset 
        # add weight for the dataset used in the training
        dataset.insert(len(dataset.columns) ,"weight", [1]*len(dataset))
        # check if the split exists, create node and recurse
        action, split_attribute = check_split(
                dataset, self.training_attributes,
                self.get_split_fn, self.decision_tree.get_attributes())
        # if split attribute does not exist then is a leaf
        if action == Actions.ADD_LEAF:
            raise SplitError(
                f"Not able to split the root node. Attributes:\n \
                Purity: {self.training_attributes.node_purity}\n \
                Minimum instances per node: {self.training_attributes.min_instances}\n \
                Maximum depth: {self.training_attributes.max_depth}")
        attr_name = split_attribute.attr_name
        attr_type = self.decision_tree.get_attributes()[attr_name]
        node_type = self.attr_type_to_decision_node[attr_type]
        threshold = None
        if split_attribute.local_threshold:
            threshold = get_total_threshold(
                    dataset[split_attribute.attr_name],
                    split_attribute.local_threshold)
        root_node_attr = DecisionNodeAttributes(
                0, "root", node_type, split_attribute.attr_name, attr_type, threshold)
        root_node = self.decision_tree.create_node(root_node_attr, None)
        self.decision_tree.add_root_node(root_node)
        self.split_fn[attr_type](root_node, dataset, split_attribute)

    def split_continuous(self,
            parent_node: Node, data_in: pd.DataFrame, split_attribute: SplitAttributes):
        """
        Recursively splits a dataset based on a continuous variable.
        decision tree adds the nodes
        """
        threshold = get_total_threshold(
                self.complete_dataset[split_attribute.attr_name], split_attribute.local_threshold)
        # change the local threshold of the parent with the total one
        parent_node.set_threshold(threshold)
        data_low = filter_dataset_low(data_in, split_attribute.attr_name, threshold)
        # check the split to know what kind of node we have to add
        action, split_attribute_low = check_split(data_low,
                self.training_attributes, self.get_split_fn, self.decision_tree.get_attributes())
        node_name = f"{parent_node.get_attribute()} <= {threshold}"
        if parent_node.get_level()+1 == self.training_attributes.max_depth:
            action = Actions.ADD_LEAF
        if action == Actions.ADD_LEAF:
            self.leaf_node_creation(parent_node, node_name, data_low)
        else:
            node, attr_type = self.node_creation(parent_node,
                    node_name, split_attribute_low)
            self.split_fn[attr_type](node, data_low, split_attribute_low)

        # Higher than the threshold
        data_high = filter_dataset_high(data_in, split_attribute.attr_name, threshold)
        # check the split to know what kind of node we have to add
        action, split_attribute_high = check_split(data_high,
                self.training_attributes, self.get_split_fn, self.decision_tree.get_attributes())
        node_name = f"{parent_node.get_attribute()} > {threshold}"
        if parent_node.get_level()+1 == self.training_attributes.max_depth:
            action = Actions.ADD_LEAF
        if action == Actions.ADD_LEAF:
            self.leaf_node_creation(parent_node, node_name, data_high)
        else:
            node, attr_type = self.node_creation(parent_node,
                    node_name, split_attribute_high)
            self.split_fn[attr_type](node, data_high, split_attribute_high)

    def split_categorical(self,
            parent_node: Node, data_in: pd.DataFrame, split_attribute: SplitAttributes):
        """
        Recursively splits a dataset based on a categorical variable.
        decision tree adds the nodes
        """
        data_known = data_in[data_in[split_attribute.attr_name] != '?']
        for attr_value in data_known[split_attribute.attr_name].unique():
            # divide data
            data = filter_dataset_cat(data_in, split_attribute.attr_name, attr_value)
            action, split_attribute_child = check_split(data,
                    self.training_attributes, self.get_split_fn,
                    self.decision_tree.get_attributes())
            # change the local threshold to total if exists
            node_name = f"{split_attribute.attr_name} = {attr_value}"
            if parent_node.get_level()+1 == self.training_attributes.max_depth:
                action = Actions.ADD_LEAF
            if action == Actions.ADD_LEAF:
                self.leaf_node_creation(parent_node, node_name, data)
            else:
                if split_attribute_child.local_threshold:
                    threshold = get_total_threshold(
                            self.complete_dataset[split_attribute_child.attr_name],
                            split_attribute_child.local_threshold)
                    split_attribute_child.threshold = threshold
                node, attr_type = self.node_creation(parent_node,
                        node_name, split_attribute_child)
                self.split_fn[attr_type](node, data, split_attribute_child)

    def node_creation(self,
            parent_node: Node, node_name: str,
            split_attribute: SplitAttributes) -> tuple[Node, AttributeType]:
        """ create the node corresponding to split attribute """
        attr_name = split_attribute.attr_name
        attr_type = self.decision_tree.get_attributes()[attr_name]
        node_type = self.attr_type_to_decision_node[attr_type]
        node_attr = DecisionNodeAttributes(
                parent_node.get_level()+1, node_name, node_type, split_attribute.attr_name,
                attr_type, split_attribute.threshold)
        node = self.decision_tree.create_node(node_attr, parent_node)
        self.decision_tree.add_node(node)
        return node, attr_type

    def leaf_node_creation(self, parent_node: Node, node_name: str, data_leaf: pd.DataFrame):
        """ create a leaf node corresponding to split attribute """
        data_leaf = data_leaf.round({'weight': 4})
        leaf_attr = LeafNodeAttributes(
                parent_node.get_level()+1, node_name, NodeType.LEAF_NODE,
                data_leaf.groupby("target")["weight"].sum().to_dict())
        node = self.decision_tree.create_node(leaf_attr, parent_node)
        self.decision_tree.add_node(node)
