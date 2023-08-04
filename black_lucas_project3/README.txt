#Project 2
--------------

## Write-Up

####Question:
What is the reason for using a decision tree stump rather than a decision tree with a greater depth? How does this differentiate adaboost from a random forest ensemble method?

####Answer:

We use decision tree stumps because depth-1 trees are faster to train and upweighting/downweighting trees of greater depths are potentially prone to overfit training data.
Adaboost weights decision stumps based on their corresponding training error, whereas random forest ensembles treat every tree of equal importance and are generally scaled past greater depths than depth-1.

####Question:
What would need to change to run an adaboost algorithm with a perceptron rather than a decision tree?

####Answer:

We would need to change each kth trained classifier from a decision tree classifier to a perceptron classifier.

### Files

`adaboost.py` - contains methods for Adaboost training and testing

### How to import

`import adaboost as ada`
