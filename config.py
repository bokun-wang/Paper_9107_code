import numpy as np

data_path = './datasets/'
plots_path = './figures/'
norm_type = np.inf
algos_path = './algorithms/'
default_it_max = 5001
is_normalized = True
eval_ratio = .01
shuffle = False
l2_weights = {'covtype': 1e-03, 'german': 1e-03, 'w8a': 1e-03, 'a9a': 1e-03, 'svmguide3': 1e-03, 'splice': 1e-03}
is_convex = False
lr_decay = False
seeds = np.random.randint(10000, size=(36, 5))
data_info = {'covtype': [581012, 54], 'german': [1000, 24], 'w8a': [49749, 300], 'a9a': [22696, 123],
             'svmguide3': [1243, 21], 'splice': [1000, 60]}
