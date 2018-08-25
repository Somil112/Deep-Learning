# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 15:20:48 2018

@author: Somil
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train=x_train.reshape(60000,784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

classifier=Sequential()
classifier.add(Dense(400,activation='relu',kernel_initializer='uniform',input_dim=784))
classifier.add(Dropout(0.3))

classifier.add(Dense(400,activation='relu',kernel_initializer='uniform'))
classifier.add(Dropout(0.3))

classifier.add(Dense(10,activation='softmax'))

classifier.summary()

classifier.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

classifier.fit(x_train,y_train,batch_size=128,epochs=20,validation_data=(x_test,y_test))

score = classifier.evaluate(x_test, y_test, verbose=0)

print(float(score[1])*100)