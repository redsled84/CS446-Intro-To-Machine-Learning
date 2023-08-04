from sklearn import tree
from math import log, exp, floor

def adaboost_train(X, Y, max_iter):
	N = len(Y)
	# initialize weights of equal importance
	weights = [1 / N for x in range(N)]

	# copy samples and labels so we don't mutate input args
	samples = [x for x in X]
	labels = [y for y in Y]

	# return lists
	f = []
	alpha = []

	# iterate max_iter times
	for k in range(max_iter):
		# train kth decision stump on "weighted" samples/labels
		_f = tree.DecisionTreeClassifier(max_depth=1)
		_f.fit(samples, labels)

		# get all predictions from kth classifier on training data
		predictions = _f.predict(X)

		# compute weighted training errror
		weighted_error = 0
		for n in range(N):
			weighted_error += weights[n] * int(Y[n] != predictions[n])

		# compute adaptive parameter
		if weighted_error == 0:
			weighted_error = 1 / 10 ** 10
		adaptive_param = 0.5 * log((1 - weighted_error) / weighted_error)

		# recompute weights
		for n in range(N):
			weights[n] *= exp(-adaptive_param * Y[n] * predictions[n])

		# normalize weights
		normalize_constant = sum(weights)
		for n in range(N):
			weights[n] /= normalize_constant

		# update alpha list with found adaptive param
		alpha.append(adaptive_param)
		# update function list with found kth classifier
		f.append(_f)

		# get inverse quantity of each weight
		dups = [floor(0.5 + 1 / weight) for weight in weights]
		# find the max value (lowest weight)
		least_prob = max(dups)

		# clear sample and labels list so we don't overpopulate them
		samples = []
		labels = []
		# iterate over the weight distribution
		for indx, prob in enumerate(dups):
			# find minimum times to duplicate each sample
			times_to_duplicate = floor(0.5 + least_prob / prob)
			# duplicate
			for times in range(times_to_duplicate):
				samples.append(X[indx])
				labels.append(Y[indx])

	# return classifiers and alpha values
	return f, alpha

def adaboost_test(X, Y, f, alpha):
	acc = 0

	# iterate over every label
	for indx, value in enumerate(Y):
		sign = 0
		# find the weighted voted prediction
		for n in range(len(f)):
			# ensure data type of each sample matches
			# expected data type for _.predict()
			strtype = str(type(X[indx]))
			prediction = 0
			# predict each sample with kth classifier
			if 'array' not in strtype or 'list' not in strtype:
				prediction = f[n].predict([X[indx]])
			else:
				prediction = f[n].predict(X[indx])
			# get summation of sign of prediction times adaptive param
			# associated with the corresponding classifier
			sign += prediction * alpha[n]

		# test if the weighted vote matches the actual value
		if sign > 0:
			acc += int(1 == value)
		elif sign < 0:
			acc += int(-1 == value)

	return round(acc / len(Y), 4)

