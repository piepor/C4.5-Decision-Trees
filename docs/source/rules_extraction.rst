Rules Extraction
================

Explainability of a decision tree is readily available right after the training. 
Given the attributes values is easy to follow the path inside the tree and to know why the model made a certain prediction.

*Rules extraction* is the process of retrieving the possible paths for every target class in the dataset.
**c4do5** has three different types of *rules extraction*, namely *Standard*, *Rules Pruning* and *Tree Pruning*. 
The pruning methods help to reduce long rules to a represetative subset.
With the function *initialize_rules_extractor*" is possible to specify the preferred method.

**Standard**

The tree is traversed from the root to the leaves. One atom of the rule consists in the conjunction of all the split conditions leading to one leaf.
The target of the atom is the class predicted by the reached leaf. 
To create the complete rule for a specific class, all the atoms with the same target are united using disjunctions.

.. code-block:: Python

   import c4dot5
   import pickle
   import pandas as pd

   training_dataset = pd.read_csv("https://raw.githubusercontent.com/piepor/C4.5-Decision-Trees/main/src/data_example/training_dataset.csv")

   with open('./example.classifier', 'rb') as file:
      decision_tree = pickle.load(file)

   rules_extractor = c4dot5.initialize_rules_extractor('standard', training_dataset, decision_tree.decision_tree)
   rules = rules_extractor.get_rules()
   rules_extractor.print_rules()

**Rules Pruning**

Sometimes the rules can become too long and unintelligible, in that case we need to prune them. 
This method is described in [J. R. Quinlan, "Simplifying Decision Trees"] and works directly on rules extracted with the standard method, dropping atoms less relevant to the classification. 

.. code-block:: Python

   import c4dot5
   import pickle
   import pandas as pd

   training_dataset = pd.read_csv("https://raw.githubusercontent.com/piepor/C4.5-Decision-Trees/main/src/data_example/training_dataset.csv")

   with open('./example.classifier', 'rb') as file:
      decision_tree = pickle.load(file)

   rules_extractor = c4dot5.initialize_rules_extractor('rules-pruning', training_dataset, decision_tree.decision_tree)
   rules = rules_extractor.get_rules()
   rules_extractor.print_rules()

**Tree Pruning**

This method is described in [J. R. Quinlan, "C4.5: Programs for Machine Learning"]. 
It prunes directly the tree replacing subtree with leaves when possible, according to the number of predicted errors. 
Then the rules are computed with the standard method on the pruned tree.

.. code-block:: Python

   import c4dot5
   import pickle
   import pandas as pd

   training_dataset = pd.read_csv("https://raw.githubusercontent.com/piepor/C4.5-Decision-Trees/main/src/data_example/training_dataset.csv")

   with open('./example.classifier', 'rb') as file:
      decision_tree = pickle.load(file)

   rules_extractor = c4dot5.initialize_rules_extractor('decision-tree-pruning', training_dataset, decision_tree.decision_tree)
   rules = rules_extractor.get_rules()
   rules_extractor.print_rules()

