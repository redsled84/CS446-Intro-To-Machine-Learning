#author: Lucas Black
#date: 12.18.22
import numpy as np
import matplotlib.pyplot as plt

def get_centered_standardized(Z):
	# compute mean and subtract from feature vectors
	Y = Z.mean(axis=0)
	centered = Z - Y

	# standardize
	X = centered.std(axis=0)
	standardized = centered / X

	return standardized

def compute_covariance_matrix(Z):
	standardized = get_centered_standardized(Z)
	
	# return Z^T * Z
	return np.dot(np.transpose(standardized), standardized)

def find_pcs(cov):
	# this function returns eigenvalues then eigenvectors
	w, v = np.linalg.eigh(cov)

	# get min args then flip to get max args
	argmins_w = np.flip(w.argsort())

	# return sorted vectors and values, respectively
	return v[argmins_w], w[argmins_w]

def project_data(Z, PCS, L):
	# project onto centered/stanardized features
	standardized = get_centered_standardized(Z)

	# project with largest eigenvector
	return np.dot(standardized, PCS[0])

def show_plot(Z, Z_star):
	standardized = get_centered_standardized(Z)

	fig, axes = plt.subplots(2)

	fig.suptitle("PCA")

	axes[0].scatter(standardized[:,0], standardized[:,1], s=10, c='red')
	axes[0].grid()
	axes[1].scatter(Z_star, np.zeros_like(Z_star), s=10, c='blue')

	plt.show()
