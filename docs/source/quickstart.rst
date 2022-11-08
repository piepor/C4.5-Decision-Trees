Quickstart
==========
This section briefly illustrates basic functions of the package.

To train a decision tree classifier, import the class DecisionTreeClassifier and call the .fit() method.
The training dataset must be a pandas DataFrame with a column named *target* to identify the target classes of the classification.

.. code-block:: python

  import pandas as pd
  from c4dot5.DecisionTreeClassifier import DecisionTreeClassifier

  training_dataset = pd.read_csv("https://raw.githubusercontent.com/piepor/C4.5-Decision-Trees/main/src/data_example/training_dataset.csv")
  attributes_map = {
    "Outlook": "categorical", "Humidity": "continuous",
    "Windy": "boolean", "Temperature": "continuous"}

  decision_tree = DecisionTreeClassifier(attributes_map)
  decision_tree.fit(training_dataset)

To make predictions, use the .predict() method

.. code-block:: python

  data_input = pd.DataFrame.from_dict({
    "Outlook": ["sunny"], "Temperature": [65], "Humidity": [90], "Windy": [False]})
  prediction = decision_tree.predict(data_input)
  print(prediction)

