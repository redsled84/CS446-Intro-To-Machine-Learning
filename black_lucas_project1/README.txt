#Decision Trees
--------------

## Write-Up

#### Question:
Did you implement your decision tree functions
iteratively or recursively? Which data structure did you
choose and were you happy with that choice? If you were
unhappy with that choice which other data structure would
have built the model with?

####Answer:
I implemented my decision tree functions `recursively` as that
was the recommended option discussed in class and it was
easier to build conceptually than an iterative design.

I choose to return a `dictionary` and I'm happy with my choice,
however, I set up my data structure to index numpy arrays based
off recursively deleted columns. Meaning, the key `col_indx` contains
a relative column index value for indexing feature data.
Another reason why I'm happy with my choice is that it is much
clearer when accessing different nodes of a decision tree given
the `left` key references a left subtree and the `right` key
references a right subtree. As opposed to a list, the keys
are more descriptive when debugging.


#### Question: 
Why might the individual trees have such variance in their
accuracy? How would you reduce this variance and potentially
improve accuracy?

#### Answer:
Individual trees may have variance in their accuracy due to
potential overfitting or feature contradictions.
Given a small random collection of the sample data during
random foresting, it's possible a trained binary decision tree
is given data which contradicts other random samples of the data.
Or it is trained where it overfits given sample data.

We can reduce this variance and potentially improve accuracy
by modifying the hyperparameter `max_depth` to find a more effective
level of accuracy.

Another way to improve accuracy would be to assign weights to
each feature node. As my current implementation treats all features
of equal importance, if there was a functional implementation
to weight the importance of features, then it could improve accuracy as well.


#### Question:
Why is it beneficial for the random forest to use an odd
number of individual trees?

#### Answer:
It's beneficial to use an odd number of individual trees with a random forest
as if we have an even number of trees, random forest can result with a binary
solution. Having an odd number of trees allows us to decide how to deal with
"inconclusive" results of a random forest.

For example, if we have 10 trees in a random forest and 5 trees have an accuracy
lower than the mean and 5 trees have an accuracy higher than the mean accuracy,
then it's essentially a flip of a coin on which subset of the random forest to apply
towards a given set of test data. Thus, adding one more tree to the random forest implies
a more reliable model.

#### Question: 
Overall, if you are still feeling uncomfortable working with
python, what aspect of the coding language do you feel you
are struggling with the most? If you do feel comfortable,
what part of python do you feel you should continue practicing?

#### Answer:
I feel fairly comfortable working with python. One part I would
like to continue practicing is file IO. I still need to reference
the web for file IO in python and it seems like an important
skill to know by heart.

Another part I would like to continue practicing is writing cleaner
python code. I wrote some horrific list comprehensions in the `data_storage.py`
file where looking back on those made me realize I probably should've modularized
those methods so they're easier to read.


### Files

`data_storage.py` - functions which hold methods for
					converting Numpy arrays into various
					data structures

`decision_trees.py` - contains methods for building/testing
					  decision trees and random forests


### Description

`data_storage.py` contains the methods

+ `build_nparray(data)`
	+ converts numpy array into numpy array suitable
	  for decision tree training
+ `build_list(data)`
	+ converts numpy array into a list
+ `build_dict(data)`
	+ converts numpy array into a dictionary, where
	  the keys for each item are the .csv headers

`decision_tree.py` contains the methods

+ `prob(x, S)`
	+ calculates the probability of the element `x` in the set `S`
+ `entropy(S)`
	+ calculates the entropy of an entire set, assuming binary feature data
+ `feature_entropy(feature_vector, labels, check_zero)`
	+ calculates the entropy on a given feature vector, provided the labels
	  and whether or not to calculate entropy on 0 or 1
+ `information_gain(total_H, feature_vector, labels)`
	+ calculates the information gain, given the entropy of an entire set
	  provided a feature vector and the list of labels
+ `new_leaf(X, Y, col_indx)`
	+ returns a data structure for a leaf in a decision tree, provided the
	  branch value and label data
+ `DT_train_binary(X, Y, max_depth)`
	+ returns a trained binary decision tree, provided the feature data, 
	label data and max depth for the tree
+ `DT_test_binary(X, Y, DT)`
	+ returns the accuracy of a given binary decision tree on feature
	data `X` and expected labels `Y`
+ `DT_make_prediction(X, DT)`
	+ returns the expected output from an input decision tree given
	  feature data `X`
+ `RF_build_random_forest(X, Y, max_depth, num_trees)`
	+ returns a list of decision trees of quantity `num_trees`. Trained on 10% of feature data
	`X`, the respective label data `Y`, at a given max depth of `max_depth`
+ `RF_test_random_forest(X, Y, RF)`
	+ outputs the accuracy of a given random forest data structure on
	random sampling of feature data `X` and label data `Y`

### How to import

`import data_storage as ds`

`import decision_tree as dt`
