from nodes import Node
from attributes import DecisionTreeAttributes, NodeAttributes, NodeType
from training import create_continuous_decision_node, create_categorical_decision_node
from training import create_boolean_decision_node, create_leaf_node


class DecisionTree:
    """ class implementing a decision tree """
    def __init__(self, attributes: DecisionTreeAttributes):
        self._nodes = set()
        self._root_node = None
        self._attributes = attributes
        self._create_node_fns = {
                NodeType.DECISION_NODE_CONTINUOUS: create_continuous_decision_node,
                NodeType.DECISION_NODE_CATEGORICAL: create_categorical_decision_node,
                NodeType.DECISION_NODE_BOOLEAN: create_boolean_decision_node,
                NodeType.LEAF_NODE: create_leaf_node,
                }

    def get_nodes(self):
        """ Returns nodes added in the tree """
        return self._nodes

    def create_node(self, node_attributes: NodeAttributes, parent_node: Node) -> Node:
        """ create a new node """
        return self._create_node_fns[node_attributes.node_type](node_attributes, parent_node)

    def add_node(self, node: Node):
        """ Add a node to the tree's set of nodes and connects it to its parent node """
        self._nodes.add(node)
        parent_node = node.get_parent_node()
        if not parent_node is None:
            parent_node.add_child(node)
        elif node.get_label() == 'root':
            self._root_node = node
        else:
            raise ValueError(f"Can't add node {node.get_label()}. \
                    Parent label not present in the tree")

    def delete_node(self, node):
        """ Removes a node from the tree's set of nodes and disconnects it from its parent node """
        parent_node = node.get_parent_node()
        if node.get_parent_node() is None:
            raise ValueError(f"Can't delete node {node.get_label()}. \
                    Parent node not found")
        parent_node.delete_child(node)
        self._nodes.remove(node)
