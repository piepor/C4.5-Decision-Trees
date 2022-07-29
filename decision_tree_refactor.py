import pandas as pd
from nodes import Node, LeafNode
from attributes import DecisionTreeAttributes, NodeAttributes, NodeType
from training import create_continuous_decision_node, create_categorical_decision_node
from training import create_leaf_node
from predictor import PredictionHandler


class DecisionTree:
    """ class implementing a decision tree """
    def __init__(self, attributes: DecisionTreeAttributes):
        self._nodes = set()
        self._root_node = None
        self._attributes = attributes
        self._create_node_fns = {
                NodeType.DECISION_NODE_CONTINUOUS: create_continuous_decision_node,
                NodeType.DECISION_NODE_CATEGORICAL: create_categorical_decision_node,
                NodeType.LEAF_NODE: create_leaf_node,
                }
        self.prediction_handler = None

    def get_root_node(self):
        """ Returns the root node of the tree """
        return self._root_node

    def get_nodes(self):
        """ Returns nodes added in the tree """
        return self._nodes

    def get_leaves_nodes(self) -> list[Node]:
        """ Returns a list of the leaves nodes """
        return {node for node in self._nodes if isinstance(node, LeafNode)}

    def create_node(self, node_attributes: NodeAttributes, parent_node: Node) -> Node:
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

    def delete_node(self, node):
        """ Removes a node from the tree's set of nodes and disconnects it from its parent node """
        parent_node = node.get_parent_node()
        parent_node.delete_child(node)
        self._nodes.remove(node)

    def predict(self, data_input: pd.DataFrame) -> list[str]:
        """ Returns the target predicted by the tree for every row in data_input """
        return self.prediction_handler.predict(data_input, self.get_root_node())
