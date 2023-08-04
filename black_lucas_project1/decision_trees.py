# Author: Lucas Black
# Date: 9.28.22
# Purpose: methods for training and testing decision trees,
#          random forest

import math
import numpy as np
import data_storage as ds

#decision tree algo:

# 1. calculate entropy of the entire set

# H(S) = sigma(c -> C) -p(c) * log^2(p(c))

# 2. Split on each feature and calculate information gain for
#    each split

# IG = H(S) - sigma(t -> T) p(t) * H(T)

# 3. Choose feature with the highest information gain

# 4. Repeat 2 - 4 until IG = 0 or IG = total entropy

# for the graduate level problems: give each feature node
# a function comparator

# get the probability of x in the set S
def prob(x, S):
    constraint = [y for y in S if y == x]

    if not len(S):
        return 0

    return len(constraint) / len(S)

# requires a 1D array (row or column)
def entropy(S):
    prob_zero = prob(0, S)
    prob_one  = prob(1, S)

    # ensure no math errors
    log_zero = 0
    if prob_zero != 0:
        log_zero = math.log(prob_zero, 2)

    log_one = 0
    if prob_one != 0:
        log_one = math.log(prob_one, 2)

    return -prob_zero * log_zero - prob_one * log_one

def feature_entropy(feature_vector, labels, check_zero):
    # we need to get the row indicies where a feature
    # data point is 1 or 0, then get the corresponding label
    # data point at that same row position and calculate entropy
    # on that set
    on_zero = np.where(feature_vector == 0)[0]
    on_one  = np.where(feature_vector == 1)[0]

    on_zero_entropy = entropy(labels[on_zero])
    on_one_entropy = entropy(labels[on_one])

    if check_zero:
        return on_zero_entropy

    return on_one_entropy

def information_gain(total_H, feature_vector, labels):
    t1 = prob(0, feature_vector) * feature_entropy(feature_vector, labels, True)
    t2 = prob(1, feature_vector) * feature_entropy(feature_vector, labels, False)
    return total_H - t1 - t2

def new_leaf(X, Y, col_indx):
    zero_indx = np.where(X[:,col_indx] == 0)[0]
    one_indx = np.where(X[:,col_indx] == 1)[0]

    if len(Y[zero_indx]) == 0:
        zero_indx = one_indx

    if len(Y[one_indx]) == 0:
        one_indx = zero_indx

    return { 'col_indx': col_indx, 'left': Y[zero_indx][0], 'right': Y[one_indx][0] }

def DT_train_binary(X, Y, max_depth):
    # check if no label data provided
    if not Y.size:
        return

    # calculate total entropy
    total_H = entropy(Y)

    # find the best information gain
    col_indx = -1
    prev_IG = -1
    column_width = X.shape[1]
    for col in range(0, column_width):
        IG = information_gain(total_H, X[:,col], Y)

        if IG > prev_IG:
            prev_IG = IG
            col_indx = col

    # check if IG = 0 or IG = total entropy
    if prev_IG == total_H or prev_IG == 0:
        return new_leaf(X, Y, col_indx)

    # after we find the best information gain,
    # then split on that feature
    if X.size > 0:
        # entropy of left branch of current feature
        left_entropy = feature_entropy(X[:,col_indx], Y, True)
        # entropy of right branch of current feature
        right_entropy = feature_entropy(X[:,col_indx], Y, False)

        left_subtree = None
        right_subtree = None

        # find corresponding feature to label indicies 
        left_indices = np.where(X[:,col_indx] == 0)[0]
        right_indices = np.where(X[:,col_indx] == 1)[0]

        # 1a. recurse down left subtree
        if left_entropy != 0:
            split = np.delete(X, col_indx, 1)
            if left_indices.size > 0:
                max_depth -= 1
                left_subtree = DT_train_binary(split[left_indices], Y[left_indices], max_depth)

        # 2a. recurse down right subtree
        if right_entropy != 0:
            split = np.delete(X, col_indx, 1)
            if right_indices.size > 0:
                max_depth -= 1
                right_subtree = DT_train_binary(split[right_indices], Y[right_indices], max_depth)

        # 1b. if no randomness in left subtree
        #     set to expected training data point
        if left_subtree is None:
            left_subtree = Y[left_indices][0]

        # 2b. if no randomness in right subtree
        #     set to expected training data point
        if right_subtree is None:
            right_subtree = Y[right_indices][0]

        # check depth
        if max_depth == 0:
            return new_leaf(X, Y, col_indx)

        # return data structure
        return {'col_indx': col_indx, 'left': left_subtree, 'right': right_subtree}


def DT_test_binary(X, Y, DT):
    acc = 0

    # go over every sample
    for indx, value in enumerate(X):
        acc += int(Y[indx] == DT_make_prediction(value, DT))
    
    return acc / X.shape[0]


def DT_make_prediction(X, DT):
    # here's the general data structure
    # I came up with:

    # {
    #     'col_indx': int,
    #     'left': int or { ... },
    #     'right': int or { ... }
    # }

    # col_indx is the column we should remove

    target_indx   = DT['col_indx']
    left_subtree  = DT['left']
    right_subtree = DT['right']

    if left_subtree is None:
        return
    if right_subtree is None:
        return

    value = X[target_indx]

    prediction = -1

    if value == 0:
        # check if there is a leaf
        if type(left_subtree) != dict:
            return left_subtree
        # continue prediction...
        else:
            split = np.delete(X, target_indx, 0)
            prediction = DT_make_prediction(split, left_subtree)
    
    if value == 1:
        # check if there is a leaf
        if type(right_subtree) != dict:
            return right_subtree
        # continue prediction...
        else:
            split = np.delete(X, target_indx, 0)
            prediction = DT_make_prediction(split, right_subtree)

    return prediction


def RF_build_random_forest(X, Y, max_depth, num_of_trees):
    RF = []

    acc = 0

    print("BUILDING RANDOM FOREST...\n")

    # create n decision trees
    for count in range(0, num_of_trees):
        # get random indicies
        random_indices = np.random.choice(X.shape[0], 
            size=math.ceil(X.shape[0] * 0.1), 
            replace=False)

        random_samples = None

        if X.shape[0] == 1:
            random_samples = X[random_indices]
        elif X.shape[0] > 1:
            random_samples = X[random_indices,:]

        random_labels  = Y[random_indices]
        dt = DT_train_binary(random_samples,  random_labels, max_depth)

        test_acc = DT_test_binary(random_samples, random_labels, dt)
        print("DT " + str(count) + " :  " + \
            str(round(test_acc, 5)))

        acc += test_acc

        RF.append(dt)

    print('RF', ':', round(acc / num_of_trees, 5), '\n')

    return RF


def RF_test_random_forest(X, Y, RF):
    if len(RF) == 0 or X.shape[1] == 0:
        return

    print("TESTING RANDOM FOREST... (" + str(len(RF)) + " trees)\n")

    # consider an individual sample
    if X.shape[1] == 1:
        acc = 0

        for count, rf in enumerate(RF):
            prediction = DT_make_prediction(X, rf)
            dt_acc = int(prediction == Y)

            print("DT " + str(count) + " :  prediction = " + str(prediction) + ", acc = " + str(dt_acc))

            acc += dt_acc

        print("RF  :  " + str(acc / len(RF)))
        return acc / len(RF)
    # consider multiple samples
    elif X.shape[1] > 1:
        acc = 0

        for count, rf in enumerate(RF):
            dt_acc = DT_test_binary(X, Y, rf)

            print("DT " + str(count) + " :  " + str(round(dt_acc, 5)))

            acc += dt_acc

        return round(acc / len(RF), 5)


# EOF
