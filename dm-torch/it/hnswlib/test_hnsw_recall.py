import numpy as np
import hnswlib

dim = 32
num_elements = 100000
k = 10
nun_queries = 10

# Generating Sample data
data = np.float32(np.random.random((num_elements, dim)))

print(data.shape)

# Declare index

hnsw_index = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
bf_index = hnswlib.Index(space='l2', dim=dim)

# Init both hnsw and brute force indices
# max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
# The capacity can be increased by saving/loading the index, see below.

