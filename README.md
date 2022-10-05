# C4.5 Decision Tree
Implementation of the Quinlan's algorithm to train a decision tree and make inference.

# Installation
```
pip install -i https://test.pypi.org/simple/ c4dot5-decision-tree
```

# Usage
To train a decision tree classifier, import the class DecisionTreeClassifier and call the .fit() method.
The training dataset must be a pandas DataFrame with a column named *target* to identify the target classes of the classification.
```python
import pandas as pd
import sklearn as sk
from c4dot5.DecisionTreeClassifier import DecisionTreeClassifier

training_dataset = pd.read_csv("https://raw.githubusercontent.com/piepor/C4.5-Decision-Trees/main/src/data_example/training_dataset.csv")
attributes_map = {
  "Outlook": "categorical", "Humidity": "continuous",
  "Windy": "boolean", "Temperature": "continuous"}

decision_tree = DecisionTreeClassifier(attributes_map)
decision_tree.fit(training_dataset)
```
To make predictions, simply use the .predict() method
data_input = pd.DataFrame.from_dict({
```python
	"Outlook": ["sunny"], "Temperature": [65], "Humidity": [90], "Windy": [False]})
prediction = decision_tree.predict(data_input)
print(prediction)
```
To visualize the decision tree use method .view(). It will show the tree and save in a folder (default to './figures')
```python
decision_tree.view(folder_name='figures', title='Quinlan-Tree')
```
To extract the splitting rules of the decision tree use the method .get_rules()
```python
rules = decision_tree.get_rules()
for rule in rules:
  print(f"{rule}: {rules[rule]}")
```

