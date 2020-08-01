import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 获取照片路径
imagepaths = []
for dirname, _, filenames in os.walk('./GTSRB_Datasets/Final_Training'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        # print(os.path.join(dirname, filename))
        if path.endswith("ppm"):
            imagepaths.append(path)
# imagepaths = imagepaths[:2000]
print(len(imagepaths))
# X for image data
X = []
# y for the labels
y = []
# Load the images into X by doing the necessary conversions and resizing of images
# Resizing is done to reduce the size of image to increase the speed of training
for path in imagepaths:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    X.append(img)
    # Getting the labels from the image path
    # ./GTSRB_Datasets/Final_Training/Images/00000
    # print(path)
    category = path.split("/")[4]
    # print(category)
    label = int(category[3:])
    # print(label)
    y.append(label)
# 把X和y转为numpy数组
X = np.array(X)
print(X.shape)
X = X.reshape(len(imagepaths), 784)
print(X.shape)
y = np.array(y)
print("Images loaded: ", len(X))
print("Labels loaded: ", len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=46)
# y_train = y_train.reshape(y_train.shape[0])
# y_test = y_test.reshape(y_test.shape[0])
y_train = keras.utils.to_categorical(y_train, 43)
y_test = keras.utils.to_categorical(y_test, 43)
# 准备x部分
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
print(X_train.shape)
print(y_train.shape)

sess = tf.InteractiveSession()


def Record_Tensor(tensor, name):
    print("Recording tensor " + name + " ...")
    f = open('./record/' + name + '.dat', 'w')
    array = tensor.eval()
    # print ("The range: ["+str(np.min(array))+":"+str(np.max(array))+"]")
    if (np.size(np.shape(array)) == 1):
        Record_Array1D(array, name, f)
    else:
        if (np.size(np.shape(array)) == 2):
            Record_Array2D(array, name, f)
        else:
            if (np.size(np.shape(array)) == 3):
                Record_Array3D(array, name, f)
            else:
                Record_Array4D(array, name, f)
    f.close()


def Record_Array1D(array, name, f):
    for i in range(np.shape(array)[0]):
        f.write(str(array[i]) + "\n")


def Record_Array2D(array, name, f):
    for i in range(np.shape(array)[0]):
        for j in range(np.shape(array)[1]):
            f.write(str(array[i][j]) + "\n")


def Record_Array3D(array, name, f):
    for i in range(np.shape(array)[0]):
        for j in range(np.shape(array)[1]):
            for k in range(np.shape(array)[2]):
                f.write(str(array[i][j][k]) + "\n")


def Record_Array4D(array, name, f):
    for i in range(np.shape(array)[0]):
        for j in range(np.shape(array)[1]):
            for k in range(np.shape(array)[2]):
                for l in range(np.shape(array)[3]):
                    f.write(str(array[i][j][k][l]) + "\n")


with tf.name_scope('input'):
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 43])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# First Convolutional Layer
with tf.name_scope('1st_CNN'):
    W_conv1 = weight_variable([3, 3, 1, 16])
    b_conv1 = bias_variable([16])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
with tf.name_scope('2rd_CNN'):
    W_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
with tf.name_scope('Densely_NN'):
    W_fc1 = weight_variable([7 * 7 * 32, 128])
    b_fc1 = bias_variable([128])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
with tf.name_scope('Softmax'):
    W_fc2 = weight_variable([128, 43])
    b_fc2 = bias_variable([43])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope('Loss'):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv+1e-8))

with tf.name_scope('Train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

tf.initialize_all_variables().run()


batch_size = 64

for i in range(5000):
    batch = np.random.choice(range(35288), size=batch_size)
    X_batch = X_train[batch]
    # print(X_batch.shape)
    y_batch = y_train[batch]

    if i % 50 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: X_batch, y_: y_batch, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: X_batch, y_: y_batch, keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: X_test, y_: y_test, keep_prob: 1.0}))

Record_Tensor(W_conv1, "W_conv1")
Record_Tensor(b_conv1, "b_conv1")
Record_Tensor(W_conv2, "W_conv2")
Record_Tensor(b_conv2, "b_conv2")
Record_Tensor(W_fc1, "W_fc1")
Record_Tensor(b_fc1, "b_fc1")
Record_Tensor(W_fc2, "W_fc2")
Record_Tensor(b_fc2, "b_fc2")
sess.close()
