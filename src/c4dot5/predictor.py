import copy
import pandas as pd
from typing import Union
from c4dot5.predicting import create_predictions_dict, select_children_for_prediction
from c4dot5.predicting import get_predictions_distribution
from c4dot5.nodes import Node, LeafNode, DecisionNode


class PredictionHandler:
    """ Takes care of the prediction part """
    def __init__(self, leaves_nodes):
        self._predictions_dict = create_predictions_dict(leaves_nodes)

    def reset_predictions(self):
        """ Resets the predictions list and predictions dictionary """
        for target in self._predictions_dict:
            self._predictions_dict[target] = []

    def _predict(self, row_input: pd.Series, node: Node):
        if node is None:
            breakpoint()
        attribute = node.get_attribute().split(":")[0]
        # in case of unknown variable more the data are passed to all children
        childs = select_children_for_prediction(row_input[attribute], node)
        for child in childs:
            if child is None:
                breakpoint()
            if isinstance(child, LeafNode):
                self._predictions_dict[child.get_label()] = child.get_classes()
            else:
                self._predict(row_input, child)

    def predict(self, data_input: pd.DataFrame, root_node: Node) -> tuple[list[str], list[dict]]:
        """ Returns the target predicted by the tree for every row in data_input """
        data_input = data_input.fillna('?')
        preds = []
        preds_distributions = []
        for _, row in data_input.iterrows():
            self.reset_predictions()
            self._predict(row, root_node)
            pred_distribution = get_predictions_distribution(self._predictions_dict)
            predicted_class = max(zip(pred_distribution.values(), pred_distribution.keys()))[1]
            preds.append(copy.copy(predicted_class))
            preds_distributions.append(copy.copy(pred_distribution))
        return preds, preds_distributions
