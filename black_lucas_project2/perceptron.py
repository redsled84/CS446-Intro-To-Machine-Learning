#Author: Lucas Black
#Date: 10.18.22

import numpy as np
import matplotlib.pyplot as plt

def perceptron_train(X, Y):
    col_size = 1
    if len(X.shape) > 1:
        col_size = X.shape[1]

    # initialize weights
    weights = np.zeros(col_size)

    # initialize bias
    bias = 0

    prev_weights = np.ones(col_size)
    # repeat until convergence
    while not np.array_equal(prev_weights, weights):
        prev_weights = np.copy(weights)
        # iterate over every sample
        for indx, value in enumerate(X):
            # compute the activation for each sample
            # (wn * xn)
            activation = weights * value

            # a = (w * x) + b
            activation = np.sum(activation) + bias

            # update rule
            if Y[indx] * activation <= 0:
                # weight updates
                for j, value in enumerate(weights):
                    weights[j] += np.sum(Y[indx] * X[indx][j])
                # update bias
                bias += Y[indx]

    return [weights, bias]


def perceptron_test(X, Y, W, B):
    acc = 0

    # compute activation for each sample
    for indx, value in enumerate(X):
        activation = (W*X[indx]).sum() + B
        # print(activation, W, X[indx], Y[indx])
        if activation > 0:
            acc += int(1 == Y[indx])
        else:
            acc += int(-1 == Y[indx])

    return acc / X.shape[0]

def writeup_plot(X, Y):
    # manually drew the decision boundry
    colors = ["#0ff0ff", "#ff0ff0", "#fae000"]
    W = perceptron_train(X, Y)

    color = None
    for indx, value in enumerate(X):
        activation = (W[0]*X[indx]).sum() + W[1]
        # print(activation, W, X[indx], Y[indx])
        if activation > 0:
            color = "#0ff0ff"
        else:
            color = "#ff0ff0"

        plt.scatter(value[0], value[1], c=color)

    plt.show()
