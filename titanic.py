# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset_train=pd.read_csv("../input/train.csv")
dataset_test=pd.read_csv("../input/test.csv")
dataset_results=pd.read_csv("../input/gender_submission.csv")
Y=dataset_results.iloc[:,:]
Y_test=dataset_results.iloc[:,1].values
X_train=dataset_train.iloc[:,[2,4,5,6,7,9,11]].values
Y_train=dataset_train.iloc[:,[1,]].values
X_test=dataset_test.iloc[:,[1,3,4,5,6,8,10]].values
X_train_frame=dataset_train.iloc[:,[2,4,5,6,7,9,11]]
Y_train_frame=dataset_train.iloc[:,[1,]]
X_test_frame=dataset_test.iloc[:,[1,3,4,5,6,8,10]]


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X_train[:, 1] = labelencoder_X1.fit_transform(X_train[:, 1])
X_test[:,1]=labelencoder_X1.transform(X_test[:,1])

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, [2,]])
X_train[:, [2,]] = imputer.transform(X_train[:, [2,]])
X_test[:, [2,]] = imputer.transform(X_test[:, [2,]])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, [2,]])

X_train_frame=pd.DataFrame(data=X_train)
X_test_frame=pd.DataFrame(data=X_test)
X_train_frame=X_train_frame.fillna('C')
X_test_frame=X_test_frame.fillna('C')
X_test_frame.loc[:,5]=X_test_frame.loc[:,5].replace('C',int(26))
X_train_frame.isna().any()
X_test_frame.isna().any()
X_train=X_train_frame.values
X_test=X_test_frame.values
Y_test=Y_test.reshape(418,1)



labelencoder_X2=LabelEncoder()
X_train[:, 6] = labelencoder_X2.fit_transform(X_train[:, 6])
X_test[:,6]=labelencoder_X2.transform(X_test[:,6 ])
onehotencoder = OneHotEncoder(categorical_features = [6])
X_train = onehotencoder.fit_transform(X_train).toarray()

X_test=onehotencoder.transform(X_test).toarray()
X_train=X_train[:,1:]
X_test=X_test[:,1:]

from sklearn.preprocessing import StandardScaler
from keras import regularizers
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier = Sequential()

classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))

classifier.add(Dropout(0.2))


classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units = 1,kernel_regularizer=regularizers.l2(0.01), kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, Y_train, batch_size = 25, epochs = 100)



y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
y_pred=y_pred*1
y_pred=y_pred.reshape(1,418)
y_pred=y_pred.flatten()

classifier.summary()
submission = pd.DataFrame({
        "PassengerId": Y["PassengerId"],
        "Survived": y_pred
    })
    
submission.to_csv('submission.csv', index=False)



