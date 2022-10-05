import pandas as pd
from typing import Union
from c4dot5.DecisionTree import DecisionTree
from c4dot5.traininghandler import TrainingHandler
from c4dot5.attributes import TrainingAttributes
from c4dot5.nodes import Node
from c4dot5.predictor import PredictionHandler
from c4dot5.visualizer import Visualizer
from c4dot5.rules_extractor import initialize_rules_extractor


class DecisionTreeClassifier:
    def __init__(
            self,
            attributes_map: dict,
            max_depth: int=10,
            node_purity: float=0.9,
            min_instances: int=2,
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
        self.decision_tree.set_prediction_handler(
                PredictionHandler(self.decision_tree.get_leaves_nodes()))

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

    def predict(self, data_input: pd.DataFrame, distribution=False) -> Union[list[str], tuple[list[str], list[dict]]]:
        """ Returns the target predicted by the tree for every row in data_input """
        predictions, predictions_distribution = self.decision_tree.predict(data_input)
        if distribution:
            return predictions, predictions_distribution
        return predictions

    def create_visualizer(self, title: str) -> Visualizer:
        visualizer = Visualizer(self.decision_tree, title)
        visualizer.create_digraph()
        return visualizer

    def view(self, title: str, folder_name: str='figures', view: bool=True):
        visualizer = self.create_visualizer(title)
        visualizer.save_view(folder_name, view=view)

    # TODO make tests
    def get_rules(self, extraction_method: str='standard', view_tree: bool=False, folder_name: str='figures') -> Union[dict, DecisionTree]:
        rules_extractor = initialize_rules_extractor(extraction_method, self.training_handler.complete_dataset, self.decision_tree)
        rules_extractor.compute()
        rules = rules_extractor.get_rules()
        if view_tree:
            visualizer = Visualizer(rules_extractor.decision_tree, f'Tree-{extraction_method}')
            visualizer.save_view(folder_name, view=True)
        return rules

