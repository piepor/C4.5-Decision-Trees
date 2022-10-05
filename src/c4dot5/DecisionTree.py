import pandas as pd
from typing import Union
from c4dot5.nodes import Node, LeafNode
from c4dot5.attributes import NodeAttributes, from_str_to_enum
from c4dot5.attributes import NodeType, AttributeType
from c4dot5.training import create_continuous_decision_node, create_categorical_decision_node
from c4dot5.training import create_leaf_node
from c4dot5.splitting import get_split_gain_continuous, get_split_gain_categorical
from c4dot5.exceptions import LeafNotFound
from c4dot5.predictor import PredictionHandler
from c4dot5.exceptions import RootNodeNotFound, PredictionHandlerNotFound


class DecisionTree:
    """ class implementing a decision tree """
    def __init__(self, attributes: dict):
        self._nodes = set()
        self._root_node = None
        self._attributes = from_str_to_enum(attributes)
        self._create_node_fns = {
                NodeType.DECISION_NODE_CONTINUOUS: create_continuous_decision_node,
                NodeType.DECISION_NODE_CATEGORICAL: create_categorical_decision_node,
                NodeType.LEAF_NODE: create_leaf_node,
                }
        self._get_split_gain_fn = {
                AttributeType.CONTINUOUS: get_split_gain_continuous,
                AttributeType.CATEGORICAL: get_split_gain_categorical,
                AttributeType.BOOLEAN: get_split_gain_categorical
                }
        self.prediction_handler = None
        self.complete_dataset = None

    def get_attributes(self) -> dict:
        """ returns the dictionary mapping data attributes and types """
        return self._attributes

    def get_root_node(self) -> Node:
        """ Returns the root node of the tree """
        if not self._root_node:
            raise RootNodeNotFound("Fit the decision tree before accessing the root node.")
        return self._root_node

    def get_nodes(self) -> Union[set[Node], set]:
        """ Returns nodes added in the tree """
        return self._nodes

    def get_leaves_nodes(self) -> set[LeafNode]:
        """ Returns a list of the leaves nodes """
        return {node for node in self._nodes if isinstance(node, LeafNode)}

    def get_leaf_node(self, leaf_label: str) -> list[Node]:
        """ Returns the leaf node with the desired label """
        if leaf_label not in {node.get_label() for node in self.get_leaves_nodes()}:
            raise LeafNotFound(f"No leaf labeled '{leaf_label}' found in the tree")
        return [node for node in self.get_leaves_nodes() if node.get_label() == leaf_label]

    def create_node(self, node_attributes: NodeAttributes, parent_node: Union[Node, None]) -> Node:
        """ create a new node """
        return self._create_node_fns[node_attributes.node_type](node_attributes, parent_node)

    def add_root_node(self, node: Node):
        """ Add a node to the tree's set of nodes and connects it to its parent node """
        self._nodes.add(node)
        self._root_node = node

    def add_node(self, node: Node):
        """ Add a node to the tree's set of nodes and connects it to its parent node """
        self._nodes.add(node)
        parent_node = node.get_parent_node()
        parent_node.add_child(node)

    def delete_node(self, node: Node):
        """ Removes a node from the tree's set of nodes and disconnects it from its parent node """
        parent_node = node.get_parent_node()
        parent_node.delete_child(node)
        self._nodes.remove(node)

    def predict(self, data_input: pd.DataFrame) -> tuple[list[str], list[dict]]:
        """ Returns the target predicted by the tree for every row in data_input """
        if not self.prediction_handler:
            raise PredictionHandlerNotFound("Fit the decision tree before predicting.")
        return self.prediction_handler.predict(data_input, self.get_root_node())

    def set_prediction_handler(self, prediciton_handler: PredictionHandler):
        self.prediction_handler = prediciton_handler
