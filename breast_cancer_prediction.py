# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 00:01:13 2018

@author: Somil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

dataset=pd.read_csv("data.csv")
X=dataset.iloc[:,2:32].values
Y=dataset.iloc[:,[1,]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y_1 = LabelEncoder()
Y[:,0] = labelencoder_Y_1.fit_transform(Y[:,0])
Y=Y.astype(float)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)

classifier = Sequential()

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, Y_train, batch_size = 25, epochs = 50,validation_data=(X_test,Y_test))

y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

#def build_classifier(optimizer,activation):
#    classifier = Sequential()
#    
#    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = activation, input_dim = 30))
#    
#    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = activation))
#    
#    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#    
#    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
#    
#    return classifier
#classifier = KerasClassifier(build_fn = build_classifier)
#parameters = {'batch_size': [25,32],
#              'epochs': [50,100],
#              'optimizer': ['adam','rmsprop','nadam'],
#              'activation':['relu','elu','tanh']
#              }
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'accuracy',cv=5
#                            )
#grid_search = grid_search.fit(X_train, Y_train)
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_
#grid_search.cv_results_