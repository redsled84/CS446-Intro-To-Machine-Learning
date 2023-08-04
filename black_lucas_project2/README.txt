#Project 2
--------------

## Write-Up

####Question:
Using the following training data provided in nearest neighbors 1.csv,
how would your algorithm classify the test points listed below with K=1, K=3, and K=5?

####Answer:

K = 1

| Test Point | Predicted Outcome |
|------------|-------------------|
| [1 1]      |         1         |
| [2 1]      |        -1         |
| [ 0 10]    |         1         |
| [10 10]    |        -1         |
| [5 5]      |         1         |
| [ 3 10]    |        -1         |
| [9 4]      |         1         |
| [6 2]      |        -1         |
| [2 2]      |         1         |
| [8 7]      |        -1         |

acc: 1.0


K = 3

| Test Point | Predicted Outcome |
|------------|-------------------|
| [1 1]      |         1         |
| [2 1]      |         1         |
| [ 0 10]    |         1         |
| [10 10]    |        -1         |
| [5 5]      |        -1         |
| [ 3 10]    |         1         |
| [9 4]      |        -1         |
| [6 2]      |         1         |
| [2 2]      |         1         |
| [8 7]      |        -1         |

acc: 0.5


K = 5

| Test Point | Predicted Outcome |
|------------|-------------------|
| [1 1]      |         1         |
| [2 1]      |         1         |
| [ 0 10]    |         1         |
| [10 10]    |        -1         |
| [5 5]      |         1         |
| [ 3 10]    |        -1         |
| [9 4]      |        -1         |
| [6 2]      |         1         |
| [2 2]      |         1         |
| [8 7]      |        -1         |

acc: 0.7

####Question:
Test your function on the following training data provided in clustering 2.csv,
with K=2 and K=3. What changes do you notice when updating the k value?

####Answer:
When K = 2 the clusters are visually split into two vertically-bound halves where
the number of data points is (approximately) equally split.

When K = 3, the cluster centers appear more in a triangle shape, where one center
is on the upper half of the plot, and two centers are on the lower half of the plot.

As well, when K = 3 the data is divided into three approximately equal sections. Where
the extra cluster when K = 3, compared to K = 2, has absorbed some of the data points from
the two clusters when K = 2.

####Question:
Test your function with K=2 and K=3 on the above data.
Plot your clusters in different colors and label the cluster centers.

####Answer:

Link to image for when K = 2,
https://imgur.com/a/uTWffuK

Link to image for when K = 3,
https://imgur.com/a/UGnqZUI

####Question:
Train your perceptron on the following dataset provided in perceptron_2.csv. Using the w and b you get, plot the decision boundary.

####Answer:

Link to image with drawn decision boundry,
https://imgur.com/a/aBKLtMg

### Files

`nearest_neighbors.py` - contains methods for building and testing
						 K-NN model

`clustering.py` - contains methods for performing K-means

`perceptron.py` - contains methods for building and testing
				  perceptron model

### How to import

`import perceptron as percept`

`import clustering as kmeans`

`import nearest_neighbors as nn`