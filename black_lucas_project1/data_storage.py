# Author: Lucas Black
# Date: 9.13.22
# Purpose: utility functions for data structures

import numpy as np

# the list comprehensions look gross but they are ・ﾟ✧ syntactic sugar ✧・ﾟ
# todo : put everything in clearly labeled functions (e.g. skipping header line,
#        last column, everything except the last column)

def build_nparray(data):
	# skip header line
	np_arr = np.array(data[1:], dtype=float)
	# get all rows minus the
	return np_arr[:np_arr.shape[0],:np_arr.shape[1] - 1], \
		np.array(np_arr[:,-1], dtype=int)

def build_list(data):
	return [list(map(float, x[:-1])) for x in data[1:]], \
		[int(x[-1]) for x in data[1:]]

def build_dict(data):
	# data[0][:-1] -> keys
	# map(float, x)) -> values [string to float]

	# [x for x in range(0, len(data) - 1)] -> keys
	# [int(y[-1]) for y in data[1:]] -> values (string to int)

	return [dict(zip(data[0][:-1], map(float, x))) for x in data[1:]], \
		dict(zip([x for x in range(0, len(data) - 1)], [int(y[-1]) for y in data[1:]]))


