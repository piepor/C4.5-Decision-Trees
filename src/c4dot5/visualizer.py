import os
from pathlib import Path
import graphviz
from c4dot5.DecisionTree import DecisionTree
from c4dot5.nodes import Node, LeafNode
from c4dot5.visualizing import create_label_leaf_node


class Visualizer:
    def __init__(self, decision_tree: DecisionTree, title: str):
        self.decision_tree = decision_tree
        self.title = title

    def create_digraph(self):
        self.dot = graphviz.Digraph(name=self.title, comment=self.title)
        root_node = self.decision_tree.get_root_node()
        label = f"{root_node.get_label()} \n [split attribute: {root_node.get_attribute()}]"
        self.dot.node(f"{root_node.get_label()}", label)
        children = sorted(list(root_node.get_children()), 
                          key=lambda x:x.get_label())
        for child in children:
            self._add_node(child, root_node)

    def _add_node(self, node: Node, parent_node: Node):
        if isinstance(node, LeafNode):
            label = create_label_leaf_node(node)
        else:
            label = f"{node.get_label()} \n [split attribute: {node.get_attribute()}]"
        self.dot.node(f"{node.get_label()}", label)
        self.dot.edge(f"{parent_node.get_label()}", f"{node.get_label()}")
        if not isinstance(node, LeafNode):
            children = sorted(list(node.get_children()), 
                              key=lambda x:x.get_label())
            for child in children:
                self._add_node(child, node)

    def save_view(self, folder_name: str, format: str='png', view: bool=True):
        curr_dir = os.getcwd()
        path = os.path.join(curr_dir, folder_name) 
        Path(path).mkdir(parents=True, exist_ok=True)
        self.dot.render(directory=path, format=format, view=view)
