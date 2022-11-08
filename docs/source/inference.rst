Inference
=========

Inference is the process of producing predictions on data given a model and a bunch of input features.

After the training, see :ref:`training`, we can retrieve the saved model using pickle.
To make predictions, use the method .predict()

.. code-block:: Python

   import pickle
   import pandas as pd

   training_dataset = pd.read_csv("https://raw.githubusercontent.com/piepor/C4.5-Decision-Trees/main/src/data_example/training_dataset.csv")

   with open('./example.classifier', 'rb') as file:
      decision_tree = pickle.load(file)

   preds = decision_tree.predict(training_dataset)

We can compute the performance of the classifier using the accuracy function from sklearn_

.. _sklearn: https://scikit-learn.org/stable/

.. code-block:: Python
  
   from sklearn import metrics

   accuracy = metrics.accuracy_score(training_dataset['target'], preds)

   print(f'Accuracy score: {accuracy}')

With **c4dot5** is possible to retrieve also the classes' distribution in the prediction.
For example, with unknown attributes, C4.5 computes the prediction as the probability of belonging to different leaves given the training instances and the others input attributes.
To return also the predictions distribution, use the parameter "*distribution*".

.. code-block:: Python

   data = pd.DataFrame.from_dict({
      "Outlook": ["sunny"], "Temperature": [70]
      "Humdity": [None], "Windy": [False]})
   preds, distr = decision_tree.predict(data, distribution=True)

   print("Prediction distribution:")
   for trg_class in distr:
      print(f"{trg_class}: {distr[trg_class]}")

