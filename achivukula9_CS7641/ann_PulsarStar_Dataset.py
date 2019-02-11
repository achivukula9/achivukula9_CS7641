# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

startExecutionTime=time.time()  

# Importing the dataset
dataset = pd.read_csv('pulsar_stars.csv')
X = dataset.iloc[:,:].values
y = dataset.iloc[:,8].values

'''
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
#required to initialize nueral network
from keras.layers import Dense #requried to build layers for nueral network

# Initialising the ANN
classifier=Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

# Adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the ANN to the Training set
classifierdata=classifier.fit(X_train,y_train,batch_size=20,epochs=100,verbose=0,validation_split=0.33)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# summarize history for accuracy
plt.plot(classifierdata.history['acc'])
plt.plot(classifierdata.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print('\n\n\n',classifierdata.history['acc'])
# summarize history for loss
plt.plot(classifierdata.history['loss'])
plt.plot(classifierdata.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print('\nTime taken to execute with best parameters ', time.time()-startExecutionTime)

print('\n\n\n',classifierdata.history['val_loss'])

# define the grid search parameters
'''
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100,150,200]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=classifier, param_grid=param_grid,scoring='accuracy',n_jobs=-1)
grid_result = grid.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
'''