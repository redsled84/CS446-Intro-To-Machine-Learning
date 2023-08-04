import numpy as np
import nearest_neighbors as nn
import clustering as kmeans
import perceptron as percept

def load_data(file_data):
    data = np.genfromtxt(file_data, skip_header=1, delimiter=',')
    X = []
    Y = []
    for row in data:
        temp = [float(x) for x in row]
        temp.pop(-1)
        X.append(temp)
        Y.append(int(row[-1]))
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

X,Y = load_data("nearest_neighbors_1.csv")
acc = nn.KNN_test(X,Y,X,Y,1)
print("KNN:", acc)

X = np.genfromtxt("clustering_1.csv", skip_header=1, delimiter=',')
mu = np.array([[1],[5]])
mu = kmeans.K_Means(X,2,mu)
print("KMeans:", mu)

X,Y = load_data("perceptron_1.csv")
W = percept.perceptron_train(X,Y)
acc = percept.perceptron_test(X,Y,W[0],W[1])
print("Percept:", acc)

# X = np.genfromtxt("clustering_3.csv", skip_header=1, delimiter=',')
# kmeans.writeup_plot(X, 3)

# X,Y = load_data("nearest_neighbors_1.csv")
# X_test = np.array([
#     [1,1],
#     [2,1],
#     [0,10],
#     [10,10],
#     [5,5],
#     [3,10],
#     [9,4],
#     [6,2],
#     [2,2],
#     [8,7]
# ])

# Y_test = np.array([
#     1,
#     -1,
#     1,
#     -1,
#     1,
#     -1,
#     1,
#     -1,
#     1,
#     -1
# ])

# print('writeup KNN')
# acc = nn.KNN_test(X, Y, X_test, Y_test, 5)
# print(acc)

# X,Y = load_data("perceptron_2.csv")
# percept.writeup_plot(X, Y)