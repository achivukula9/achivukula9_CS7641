# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import time
startExecutionTime=time.time()

# Importing the dataset
dataset = pd.read_csv('pulsar_stars.csv')
X = dataset.iloc[:,:].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Feature Scaling
#for decision tree, we needn't do feature scaling as it's not euclidean. Plotting has high resolution so it does some scaling to run faster
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

criterion=['gini','entropy']
splitter=['best','random']
max_depth=[1,2,3,4,5,6,7,8,9,10]
min_samples_split=[2,3,4,5,0.1,0.3,0.6,1.0]
min_weight_fraction_leaf=[0,0.2,0.3,0.4]
#max_features=[1,2,3,4,5,1.0,2.0,3.0,4.0,'None','auto']

hyperparameters=dict(criterion=criterion,splitter=splitter,max_depth=max_depth,min_samples_split=min_samples_split,min_weight_fraction_leaf=min_weight_fraction_leaf)

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV



gridSearch=GridSearchCV(DecisionTreeClassifier(),hyperparameters,cv=5,n_jobs=-1)
gridResults=gridSearch.fit(X_train,y_train)

print("Best Accuracy Score %f\n Best Parameters %s\n Best Splits %i" %(gridResults.best_score_,gridResults.best_params_,gridResults.n_splits_ ))
print('\nTime taken to execute with best parameters ', time.time()-startExecutionTime)




#Fitting classifier to the Training set

classifier=DecisionTreeClassifier(criterion='gini',max_depth=1,min_samples_split=2,min_weight_fraction_leaf=0.0,splitter='best')
classifier.fit(X_train,y_train)


 #Predicting the Test set results
y_pred = classifier.predict(X_test)

print('\n Execution Start Time ',startExecutionTime)
print('\n current Time ',time.time())
print('\nTime taken to execute with best parameters ', time.time()-startExecutionTime)



print('\n\n\n\n',classification_report(y_test,y_pred))

 #Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

##Get accuracy score
accuracyScore=accuracy_score(y_test,y_pred)
print('\n\n\n\n', 'accuracy score is : ',accuracyScore)

f1Score=f1_score(y_test,y_pred,average=None)
print('\n\n\n\n',' f1 score is : ',f1Score)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve

#Coming up with training sizes
train_sizes=[1,50,100,150,200,250,300,4000,8000]


train_sizes, training_scores, test_scores = learning_curve(DecisionTreeClassifier(criterion='gini',max_depth=1,min_samples_split=2,min_weight_fraction_leaf=0.0,splitter='best'),
                                                   X,
                                                   y, train_sizes = train_sizes, cv = 5,
                                                   scoring = 'neg_mean_squared_error',shuffle='True')

print('\n\n\nTraining scores:\n\n', training_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', test_scores)

training_scores_mean = -training_scores.mean(axis = 1)
test_scores_mean = -test_scores.mean(axis = 1)

print('\n\nMean training scores\n\n', pd.Series(training_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean test scores\n\n',pd.Series(test_scores_mean, index = train_sizes))


plt.style.use('seaborn')

plt.plot(train_sizes, training_scores_mean, label = 'Training error')
plt.plot(train_sizes, test_scores_mean, label = 'Test error')

plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a pruned decision tree with hyperparametrized classification', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,0.5)
plt.show()



train_sizes, train_scores, test_scores = learning_curve(DecisionTreeClassifier(criterion='gini',max_depth=1,min_samples_split=2,min_weight_fraction_leaf=0.0,splitter='best'),
                                                   X,
                                                   y, train_sizes = train_sizes, cv = 5,
                                                   scoring = 'accuracy',shuffle='True')


# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()




