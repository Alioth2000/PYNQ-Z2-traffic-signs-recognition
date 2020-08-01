# Normal Model
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout


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
    #print(category)
    label = int(category[3:])
    # print(label)
    y.append(label)
# 把X和y转为numpy数组
X = np.array(X)
print(X.shape)
X = X.reshape(len(imagepaths), 28, 28, 1)
print(X.shape)
y = np.array(y)
print("Images loaded: ", len(X))
print("Labels loaded: ", len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])
# y_train = keras.utils.to_categorical(y_train, 43)
# y_test = keras.utils.to_categorical(y_test, 43)
# 准备x部分
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
print(X_train.shape)
print(y_train.shape)
model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Dense(128, activation='relu'))

model.add(Flatten())

model.add(Dropout(0.3))

model.add(Dense(43, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=2, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test)
print(score[0], score[1])

model.save('./model/GTSRB.h5')
json_string = model.to_json()
with open("./model/GTSRB.json", "w") as f:
    f.write(json_string)
