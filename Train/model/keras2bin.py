import h5py
from keras.models import load_model
import numpy as np


model = load_model('./GTSRB.h5')	#模型导入
f = h5py.File('./GTSRB.h5', 'r')  # 打开h5文件
for key in f.keys():  # 查看内部的键
    print(key)
print(f['model_weights'].attrs.keys())  # 查看键的属性
print(f['model_weights'].attrs['layer_names'])  # 查看层的名称
layer=['conv2d_1', 'conv2d_2', 'conv2d_3', 'dense_1']
for i in layer:
    weight, bias = model.get_layer(i).get_weights()
    weight = weight.flatten()
    bias = bias.flatten()

    # save as txt
    w_name = str(i) + '_w' + '.txt'
    np.savetxt(w_name, weight, fmt='%f')
    b_name = str(i) + '_b' + '.txt'
    np.savetxt(b_name, bias, fmt='%f')

    # save as bin
    w_name = str(i) + '_w' + '.bin'
    weight.tofile(w_name)
    b_name = str(i) + '_b' + '.bin'
    bias.tofile(b_name)
