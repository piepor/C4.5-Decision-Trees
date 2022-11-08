.. _training:

Training
========

Training is a fundamental phase in a machine learning pipline where the model learns from data to achieve the required task. 
Considering a decision tree, the task is to classify the output type of an object given some input features.

**c4dot5** provides a training procedure implementing the algorithm described in Quinlan's book "C4.5: Programs for machine learning".
The dataset must be a Pandas_ dataframe with a column named "*target*" identifying the desired output class.

.. _Pandas: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html


The C4.5 algorithm admits three types of input features

  - *continuous* : variables belonging to the set of real numbers (e.g. "*amount of money*")
  - *boolean* : variables taking only two values, *True* or *False*
  - *categorical* : variables taking two or more values (e.g. "*pizza*" could be "*Margherita*", "*Capricciosa*", "*Diavola*" ...)
It is important to specify the right type since they are treated differently. In **c4dot5** implementation the feature type must be specified in the so-called "*attributes map*" as follows:

.. code-block:: Python

  import pandas as pd
  from c4dot5.DecisionTreeClassifier import DecisionTreeClassifier

  training_dataset = pd.read_csv("https://raw.githubusercontent.com/piepor/C4.5-Decision-Trees/main/src/data_example/training_dataset.csv")
  attributes_map = {
    "Outlook": "categorical", "Humidity": "continuous",
    "Windy": "boolean", "Temperature": "continuous"}

The "*attributes map*" is simply a python dictionary relating the feature name (**must** be the same name of the dataframe column) with its attribute type. 

The training algorithm use a *divide and conquer* startegy, dividing the the dataset in sub-dataset each containing rows with specific characteristics (e.g. *attribute_1 < 100*). 
The dataset is complete at the *root* node while *leaves* contain the final decomposed sub-datasets. The splitting attribute is chosen maximixing at each level the information gain with respect to the sub-dataset considered. 
Every dataset split corresponds to a node in the decision tree that, consequently, has a hierachical structure. 

**c4dot5** has three parameters controlling the tree growth and, therefore, its representative power. In fact, deeper trees tend to overfit the data while too shallow tree risk to underfit them.
The three main control parameters are

  - *node_purity* (default=0.9): is the minimum fraction of the maximum class in a leaf 
  - *max_depth* (default=10): maximum *number of levels* of the tree
  - *min_instances* (default=2): minimum *number of instances* inside a leaf
The parameters are set during the initialization of the DecisionTreeClassifier class.

.. code-block:: Python
      
  decision_tree = DecisionTreeClassifier(attributes_map,
        node_purity=0.9, max_depth=10, min_instances=2)

Once the classifier is instantiated, it can be trained using the method .fit().
After the training, we can save the model in *json* format with the method .save() specifyng the output file name and path.

.. code-block:: Python

  decision_tree.fit(training_dataset)
  decision_tree.save('./example.classifier')

To evaluate the splits and choose the best one, C4.5 uses the entropy function. 
In **c4dot5** the function is customizable. To write a compatible function, its input *must* be a dataset with two columns ('target' and 'weigths') while the output *must* be of type *float*.
For example, the default function is the following:

.. code-block:: Python

  def class_entropy(data) -> float:
      """ Returns the weighted entropy of a split """
      ops = data.groupby('target')['weight'].sum() / data['weight'].sum()
      return - np.sum(ops * np.log2(ops))

  decision_tree = DecisionTreeClassifier(
        attributes_map, node_purity=0.9, max_depth=10,
        min_instances=2, evaluate_split_fn=class_entropy)

