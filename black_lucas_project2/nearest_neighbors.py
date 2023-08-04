#Author: Lucas Black
#Date: 10.18.22

import numpy as np
from scipy.spatial.distance import euclidean

def KNN_predict(train_input, X, Y, K):
    # store the distances between the input
    # vector and the training data
    distances = []
    for indx, value in enumerate(X):
        distances.append([
            euclidean(value, train_input), # euclidean distance
            Y[indx]                        # label
        ])

    # sort by the first element in each sublist
    distances.sort(key = lambda x: x[0])
    # take the first K samples
    distances = distances[:K]
    # grab only the signs from the resulting list
    signs = [y[1] for y in distances]

    if sum(signs) > 0:
        return 1

    return -1


def KNN_test(X_train, Y_train, X_test, Y_test, K):
    acc = 0
    # iterate over every test sample
    for indx, value in enumerate(X_test):
        # get a prediction
        prediction = KNN_predict(value, X_train, Y_train, K)
        acc += int(prediction == Y_test[indx])
    return acc / Y_test.size

