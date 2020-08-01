import numpy as np

file_name = ['b_conv1', 'b_conv2', 'b_fc1', 'b_fc2', 'w_conv1', 'w_conv2', 'w_fc1',
             'w_fc2']
for i in file_name:
    to_bin = np.loadtxt('./' + i + '.dat', dtype='float32')
    print(to_bin.dtype)
    print(to_bin.shape)
    to_bin.tofile('./' + i + '.bin')
