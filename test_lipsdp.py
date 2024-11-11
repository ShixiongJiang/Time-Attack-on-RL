import numpy as np
import os
weights = []
net_dims = [2, 10, 30, 20, 2]
num_layers = len(net_dims) - 1
norm_const = 1 / np.sqrt(num_layers)

for i in range(1, len(net_dims)):
  weights.append(norm_const * np.random.rand(net_dims[i], net_dims[i-1]))


from scipy.io import savemat

fname = os.path.join(os.getcwd(), 'LipSDP/LipSDP/examples/saved_weights/random_weights.mat')
data = {'weights': np.array(weights, dtype=object)}
savemat(fname, data)
