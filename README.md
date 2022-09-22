# C4.5-Decision-Tree
Implementation of the Quinlan's algorithm to train a decision tree and make inference.

# Usage
To train a decision tree classifier, import the class DecisionTreeClassifier and call the .fit() method.
```python
import pandas as pd
from c4dot5.DecisionTreeClassifier import DecisionTreeClassifier

training_dataset = pd.read_csv()

decision_tree = DecisionTreeClassifier(attributes_map)
decision_tree.fit(training_dataset)
```
To make predictions, simply use the .predict() method
```python
data_input = pd.DataFrame.from_dict({
	"Outlook": ["sunny"], "Temperature": [65], "Humidity": [90], "Windy": [False]})
prediction = decision_tree.predict(data_input)
print(prediction)
```
