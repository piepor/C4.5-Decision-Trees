from __future__ import annotations
from copy import copy
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
from c4dot5.nodes import LeafNode, DecisionNode
from c4dot5.DecisionTree import DecisionTree
from c4dot5.attributes import LeafNodeAttributes, NodeType
from statsmodels.stats.proportion import proportion_confint


def extract_rules_from_leaf(node: LeafNode) -> list[str]:
    """ Extract all the rules in the path from the leaf to the root """
    rules = list()
    current_node = copy(node)
    while current_node.get_label() != 'root':
        rules.append(current_node.get_label())
        current_node = current_node.get_parent_node()
    return rules

def standard_extraction(decision_tree: DecisionTree) -> dict:
    """ Extracts the rules from the tree, one for each target transition.

    For each leaf node, puts in conjunction all the conditions in the path from the root to the leaf node.
    Then, for each target class, put the conjunctive rules in disjunction.
    """
    rules = dict()
    leaf_nodes = decision_tree.get_leaves_nodes()
    for leaf_node in leaf_nodes:
        vertical_rules = extract_rules_from_leaf(leaf_node)

        vertical_rules = ' && '.join(vertical_rules)

        if leaf_node.get_class_name() not in rules.keys():
            rules[leaf_node.get_class_name()] = set()
        rules[leaf_node.get_class_name()].add(vertical_rules)

    for target_class in rules.keys():
        rules[target_class] = ' || '.join(rules[target_class])

    return rules

def extract_rules_with_pruning(dt, data_in) -> dict:
    """ Extracts the rules from the tree, one for each target transition.

    For each leaf node, takes the list of conditions from the root to the leaf, simplifies it if possible, and puts
    them in conjunction, adding the resulting rule to the dictionary at the corresponding target class key.
    Finally, all the rules related to different leaves with the same target class are put in disjunction.
    """

    # Starting with a p_value threshold for the Fisher's Exact Test (for rule pruning) of 0.01, create the rules
    # dictionary. If for some target all the rules have been pruned, repeat the process increasing the threshold.
    p_threshold = 0.01
    keep_rule = set()
    while p_threshold <= 1.0:
        rules = dict()

        leaves = dt.get_leaves_nodes()
        inputs = [(leaf, keep_rule, p_threshold, data_in) for leaf in leaves]
        print("Starting multiprocessing rules pruning on {} leaves...".format(str(len(leaves))))
        with multiprocessing.Pool() as pool:
            result = list(tqdm(pool.imap(_simplify_rule_multiprocess, inputs), total=len(leaves)))

        for (vertical_rules, leaf_class) in result:
            # Create the set corresponding to the target class in the rules dictionary, if not already present
            if leaf_class not in rules.keys():
                rules[leaf_class] = set()

            # If the resulting list is composed by at least one rule, put them in conjunction and add the result to
            # the dictionary of rules, at the corresponding class label
            if len(vertical_rules) > 0:
                vertical_rules = " && ".join(vertical_rules)
                rules[leaf_class].add(vertical_rules)

        # Put the rules for the same target class in disjunction. If there are no rules for some target class (they
        # have been pruned) then set the 'empty_rule' variable to True.
        empty_rule = False
        for target_class in rules.keys():
            if len(rules[target_class]) == 0:
                empty_rule = True
                break
            else:
                rules[target_class] = " || ".join(rules[target_class])

        # If 'empty_rule' is True, then increase the threshold and repeat the process. Otherwise, if two target
        # transitions have the same rule (because the original vertical rule has been pruned "too much"), repeat the
        # process but avoid simplifying the rule that originated the problem. This is done only if the 'new' rules
        # to be avoided are not all already present in 'keep_rule', otherwise it means that the process is looping.
        # If that happens, simply increase the threshold and repeat the process.
        # Otherwise, return the dictionary.
        if empty_rule:
            # TODO maybe increase more each time? This is precise but it may take long since the cap is 1
            keep_rule = set()
            p_threshold = round(p_threshold + 0.01, 2)
        elif len(rules.values()) != len(set(rules.values())):
            rules_to_add = [r for r in set(rules.values()) if list(rules.values()).count(r) > 1]
            if not all([r in keep_rule for r in rules_to_add]):
                keep_rule.update(rules_to_add)
            else:
                keep_rule = set()
                p_threshold = round(p_threshold + 0.01, 2)
        else:
            break

    return rules


def _simplify_rule_multiprocess(input):
    leaf_node, kr, p_threshold, data_in = input
    vertical_rules = extract_rules_from_leaf(leaf_node)

    # Simplify the list of rules, if possible (and if vertical_rules does not contain rules in keep_rule)
    if not any([r in vertical_rules for r in kr]):
        vertical_rules = _simplify_rule(vertical_rules, leaf_node._label_class, p_threshold, data_in)

    return vertical_rules, leaf_node._label_class


def _simplify_rule(vertical_rules, leaf_class, p_threshold, data_in) -> list:
    """ Simplifies the list of rules from the root to a leaf node.

    Given the list of vertical rules for a leaf, i.e. the list of rules from the root to the leaf node,
    drops the irrelevant rules recursively applying a Fisher's Exact Test and returns the remaining ones.
    In principle, all the rules could be removed: in that case, the result would be an empty list.
    Method taken from "Simplifying Decision Trees" by J.R. Quinlan (1986).
    """

    rules_candidates_remove = list()
    # For every rule in vertical_rules, check if it could be removed from vertical_rules.
    # This is true if the related p-value returned by the Fisher's Exact Test is higher than the threshold.
    # Indeed, a rule is considered relevant for the classification only if the null hypothesis (i.e. the two
    # variables - table rows and columns - are independent) can be rejected at the threshold*100% level or better.
    for rule in vertical_rules:
        other_rules = vertical_rules[:]
        other_rules.remove(rule)
        table = _create_fisher_table(rule, other_rules, leaf_class, data_in)
        (_, p_value) = stats.fisher_exact(table)
        if p_value > p_threshold:
            rules_candidates_remove.append((rule, p_value))

    # Among the candidates rules, remove the one with the highest p-value (the most irrelevant)
    if len(rules_candidates_remove) > 0:
        rule_to_remove = max(rules_candidates_remove, key=itemgetter(1))[0]
        vertical_rules.remove(rule_to_remove)
        # Then, recurse the process on the remaining rules
        _simplify_rule(vertical_rules, leaf_class, p_threshold, data_in)

    return vertical_rules


def _create_fisher_table(rule, other_rules, leaf_class, data_in) -> pd.DataFrame:
    """ Creates a 2x2 table to be used for the Fisher's Exact Test.

    Given a rule from the list of rules from the root to the leaf node, the other rules from that list, the leaf
    class and the training set, creates a 2x2 table containing the number of training examples that satisfy the
    other rules divided according to the satisfaction of the excluded rule and the belonging to target class.
    Missing values are not taken into account.
    """

    # Create a query string with all the rules in "other_rules" in conjunction (if there are other rules)
    # Get the examples in the training set that satisfy all the rules in other_rules in conjunction
    if len(other_rules) > 0:
        query_other = ""
        for r in other_rules:
            r_attr, r_comp, r_value = r.split(' ')
            query_other += r_attr
            if r_comp == '=':
                query_other += ' == '
            else:
                query_other += ' ' + r_comp + ' '
            if data_in.dtypes[r_attr] in ['float64', 'bool']:
                query_other += r_value
            else:
                query_other += '"' + r_value + '"'
            if r != other_rules[-1]:
                query_other += ' & '
        examples_satisfy_other = data_in.query(query_other)
    else:
        examples_satisfy_other = data_in.copy()

    # Create a query with the excluded rule
    rule_attr, rule_comp, rule_value = rule.split(' ')
    query_rule = rule_attr
    if rule_comp == '=':
        query_rule += ' == '
    else:
        query_rule += ' ' + rule_comp + ' '
    if data_in.dtypes[rule_attr] in ['float64', 'bool']:
        query_rule += rule_value
    else:
        query_rule += '"' + rule_value + '"'

    # Get the examples in the training set that satisfy the excluded rule
    examples_satisfy_other_and_rule = examples_satisfy_other.query(query_rule)

    # Get the examples in the training set that satisfy the other_rules in conjunction but not the excluded rule
    examples_satisfy_other_but_not_rule = examples_satisfy_other[
        ~examples_satisfy_other.apply(tuple, 1).isin(examples_satisfy_other_and_rule.apply(tuple, 1))]

    # Create the table which contains, for every target class and the satisfaction of the excluded rule,
    # the corresponding number of examples in the training set
    table = {k1: {k2: 0 for k2 in [leaf_class, 'not '+leaf_class]} for k1 in ['satisfies rule', 'does not satisfy rule']}

    count_other_and_rule = examples_satisfy_other_and_rule.groupby('target').count().iloc[:, 0]
    count_other_but_not_rule = examples_satisfy_other_but_not_rule.groupby('target').count().iloc[:, 0]

    for idx, value in count_other_and_rule.items():
        if idx == leaf_class:
            table['satisfies rule'][leaf_class] = value
        else:
            table['satisfies rule']['not '+leaf_class] = value
    for idx, value in count_other_but_not_rule.items():
        if idx == leaf_class:
            table['does not satisfy rule'][leaf_class] = value
        else:
            table['does not satisfy rule']['not '+leaf_class] = value

    table_df = pd.DataFrame.from_dict(table, orient='index')
    return table_df

def pessimistic_pruning(decision_tree: DecisionTree, data_in: pd.DataFrame) -> None:
    """ Prunes the decision tree, substituting subtrees with leaves when possible.

    Given a subtree of the decision tree, computes the number of predicted errors keeping the subtree as it is.
    Then, for each target value among its leaves, computes the number of predicted errors substituting the subtree
    with a leaf having that target value. If the number of predicted errors after the substitution is lower than
    the one before the substitution, then replaces the subtree with a leaf having as target value the one which
    gave the smallest number of predicted errors.
    The procedure is repeated for every subtree in the decision tree, starting from the bottom. Every time the
    decision tree is pruned, the method is called recursively on the pruned decision tree in order to re-evaluate it
    for possibly further pruning.

    Method is taken by "Pruning Decision Trees" in "C4.5: Programs for Machine Learning" by J. R. Quinlan (1993).
    """

    decision_tree = copy(decision_tree)
    subtrees_to_prune = set()
    subtrees_with_only_leaves = [node for node in decision_tree.get_nodes() if isinstance(node, DecisionNode) and
                                 node.get_label() != 'root' and
                                 all(isinstance(child, LeafNode) for child in node.get_children())]

    for subtree in subtrees_with_only_leaves:
        target_values = set()
        subtree_errors = 0
        # Number of predicted errors of the subtree
        for leaf in subtree.get_children():
            subtree_errors += _compute_number_predicted_errors(leaf, leaf.get_class_name(), data_in)
            target_values.add(leaf.get_class_name())

        # Number of predicted errors replacing the subtree with a leaf for every target value among its children
        tests_results = dict()
        for target_value in target_values:
            tests_results[target_value] = _compute_number_predicted_errors(subtree, target_value, data_in)

        # Storing the subtree if it needs to be pruned
        min_tests_item = min(tests_results.items(), key=lambda x: x[1])
        # TODO added or conditions: in case of 'nan', then prune
        if min_tests_item[1] < subtree_errors or np.isnan(subtree_errors) or np.isnan(min_tests_item[1]):
            subtrees_to_prune.add((subtree, min_tests_item[0]))

    # Pruning the stored subtrees
    for subtree, target_value in subtrees_to_prune:
        data_leaf = data_in.round({'weight': 4})
        leaf_attr = LeafNodeAttributes(
                subtree.get_level(), subtree.get_label(), NodeType.LEAF_NODE,
                #{target_value: len(data_leaf[data_leaf['target'] == target_value])})
                # New version considering the weight
                {target_value: data_leaf.query("target == @target_value")["weight"].sum()})
        node = decision_tree.create_node(leaf_attr, subtree.get_parent_node())
        decision_tree.add_node(node)

    # Recursion only if at least one subtree has been pruned
    if subtrees_to_prune:
        pessimistic_pruning(decision_tree, data_in)


def _compute_number_predicted_errors(node, target_value, data_in) -> float:
    """ Computes the number of predicted errors given a node, a target value and the training set.

    First, it computes the number N of training instances covered by that node of the decision tree. Then, it
    isolates the ones E that have a target value different from the one prescribed by the node. Finally, it computes
    tha number of predicted errors of that node. This is computed as the product between N and the upper bound of a
    binomial distribution with N trials and E observed events.
    """

    branch_conditions = extract_rules_from_leaf(node)

    query = ""
    for r in branch_conditions:
        r_attr, r_comp, r_value = r.split(' ')
        query += r_attr
        if r_comp == '=':
            query += ' == '
        else:
            query += ' ' + r_comp + ' '
        if data_in.dtypes[r_attr] in ['float64', 'bool']:
            query += r_value
        else:
            query += '"' + r_value + '"'
        if r != branch_conditions[-1]:
            query += ' & '

    node_instances = data_in.query(query)
    wrong_instances = node_instances[node_instances['target'] != target_value]

    # TODO Sometimes both len are 0 so the upper bound is 'nan' (this also raises a warning since division by 0)
    return len(node_instances) * proportion_confint(len(wrong_instances), len(node_instances), method='beta', alpha=0.50)[1]
