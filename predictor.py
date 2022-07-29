import copy
import pandas as pd
from predicting import create_predictions_dict, select_childs_for_prediction
from nodes import Node, LeafNode
from predicting import get_predictions_distribution


class PredictionHandler:
    """ Takes care of the prediction part """
    def __init__(self, leaves_nodes):
        self._predictions_dict = create_predictions_dict(leaves_nodes)
        self._predictions = []
        self._total_sum = 0

    def reset_predictions(self):
        """ Resets the predictions list and predictions dictionary """
        for target in self._predictions_dict:
            self._predictions_dict[target] = []
        self._total_sum = 0
        self._predictions = []

    def _predict(self, row_input: pd.Series, node: Node):
        attribute = node.get_attribute().split(":")[0]
        childs = select_childs_for_prediction(row_input[attribute], node)
        for child in childs:
            if isinstance(child, LeafNode):
                for target in child.get_classes():
                    self._predictions_dict[target].append(child.get_classes()[target])
            else:
                self._predict(row_input, child)

    def predict(self, data_input: pd.DataFrame, root_node: Node) -> list[str]:
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
