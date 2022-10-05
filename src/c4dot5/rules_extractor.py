from __future__ import annotations
from abc import ABC, abstractmethod
from copy import copy
import pandas as pd
from c4dot5.DecisionTree import DecisionTree
from c4dot5.extracting_rules import standard_extraction, pessimistic_pruning, extract_rules_with_pruning
from c4dot5.exceptions import RulesNotFound


#TODO write tests
def initialize_rules_extractor(extraction_method: str, dataset: pd.DataFrame, decision_tree: DecisionTree) -> ExtractionMethod:
    if extraction_method == 'standard':
        return StandardExtraction(decision_tree)
    if extraction_method == 'decision-tree-pruning':
        return DecisionTreePruning(decision_tree, dataset)
    if extraction_method == 'rules-pruning':
        return RulesPruning(decision_tree, dataset)


class ExtractionMethod(ABC):
    @abstractmethod
    def compute(self):
        """ extract the rules from a decision tree """

    @abstractmethod
    def get_rules(self) -> dict:
        """ extract the rules from a decision tree """

class StandardExtraction(ExtractionMethod):
    def __init__(self, decision_tree: DecisionTree):
        self.decision_tree = copy(decision_tree)
        self._rules = None

    def compute(self):
        self._rules = standard_extraction(self.decision_tree)

    def get_rules(self) -> dict:
        if not self._rules:
            raise RulesNotFound("Rules not found. Pleas compute the rules before accessing them.")
        return self._rules

class DecisionTreePruning(ExtractionMethod):
    def __init__(self, decision_tree: DecisionTree, dataset: pd.DataFrame):
        self.decision_tree = copy(decision_tree)
        self.dataset = copy(dataset)
        self._rules = None

    def compute(self):
        self.decision_tree = pessimistic_pruning(self.decision_tree, self.dataset)
        self._rules = standard_extraction(self.decision_tree)

    def get_rules(self) -> dict:
        if not self._rules:
            raise RulesNotFound("Rules not found. Pleas compute the rules before accessing them.")
        return self._rules

class RulesPruning(ExtractionMethod):
    def __init__(self, decision_tree: DecisionTree, dataset: pd.DataFrame):
        self.decision_tree = copy(decision_tree)
        self.dataset = copy(dataset)
        self._rules = None

    def compute(self):
        self._rules = extract_rules_with_pruning(self.decision_tree, self.dataset)

    def get_rules(self) -> dict:
        if not self._rules:
            raise RulesNotFound("Rules not found. Pleas compute the rules before accessing them.")
        return self._rules
