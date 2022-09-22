import pandas as pd
from DecisionTree import DecisionTree
from traininghandler import TrainingHandler
from attributes import TrainingAttributes
from nodes import Node


class DecisionTreeClassifier:
    def __init__(
            self,
            attributes_map: dict,
            max_depth: int=10,
            node_purity: float=0.9,
            min_instances: int=2
            ):
        self.decision_tree = DecisionTree(attributes_map)
        training_attributes = TrainingAttributes(
                max_depth=max_depth,
                node_purity=node_purity,
                min_instances=min_instances)
        self.training_handler = TrainingHandler(
                self.decision_tree,
                training_attributes)

    def fit(self, dataset: pd.DataFrame):
        """ fit the input dataset """
        self.training_handler.split_dataset(dataset)

    def get_attributes(self) -> dict:
        """ returns the dictionary mapping data attributes and types """
        return self.decision_tree.get_attributes()

    def get_root_node(self):
        """ Returns the root node of the tree """
        return self.decision_tree.get_root_node()

    def get_nodes(self):
        """ Returns nodes added in the tree """
        return self.decision_tree.get_nodes()

    def get_leaves_nodes(self) -> set[Node]:
        """ Returns a list of the leaves nodes """
        return self.decision_tree.get_leaves_nodes()

    def get_leaf_node(self, leaf_label: str) -> list[Node]:
        """ Returns the leaf node with the desired label """
        return self.decision_tree.get_leaf_node(leaf_label)

    def predict(self, data_input: pd.DataFrame) -> list[str]:
        """ Returns the target predicted by the tree for every row in data_input """
        return self.decision_tree.predict(data_input, self.get_root_node())
