# Load libraries
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
startExecutionTime=time.time()


# Load data
dataset = pd.read_csv('covtype.csv')
X = dataset.iloc[:,:].values
y = dataset.iloc[:, 54].values

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=10,
                         learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

n_estimators=[1,2,5,10,20,30,40,50]
learning_rate=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0]
algorithm=['SAMME','SAMME.R']


hyperparameters=dict(n_estimators=n_estimators,learning_rate=learning_rate,algorithm=algorithm)

gridSearch=GridSearchCV(AdaBoostClassifier(),hyperparameters,cv=5,n_jobs=-1)
gridResults=gridSearch.fit(X_train,y_train)

print("\n\nBest Accuracy Score %f\n Best Parameters %s\n Best Splits %i" %(gridResults.best_score_,gridResults.best_params_,gridResults.n_splits_ ))



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




#Coming up with training sizes
train_sizes=[5,50,100,150,200,250,300,4000,8000]

#Features=['Mean of the integrated profile',' Standard deviation of the integrated profile',' Excess kurtosis of the integrated profile',' Skewness of the integrated profile',' Mean of the DM-SNR curve',' Standard deviation of the DM-SNR curve',' Excess kurtosis of the DM-SNR curve',' Skewness of the DM-SNR curve']
#target='target_class'

train_sizes, training_scores, test_scores = learning_curve(AdaBoostClassifier(n_estimators = 1,learning_rate=1,algorithm='SAMME' ),
                                                   X,
                                                   y, train_sizes = train_sizes, cv = 5,
                                                   scoring = 'neg_mean_squared_error',shuffle='True')

print('Training scores:\n\n', training_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', test_scores)

training_scores_mean = -training_scores.mean(axis = 1)
test_scores_mean = -test_scores.mean(axis = 1)

print('Mean training scores\n\n', pd.Series(training_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean test scores\n\n',pd.Series(test_scores_mean, index = train_sizes))


plt.style.use('seaborn')

plt.plot(train_sizes, training_scores_mean, label = 'Training error')
plt.plot(train_sizes, test_scores_mean, label = 'Test error')

plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for pruned Adaboost random classification parameters', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,3)
plt.show()

#print('\nTime taken to execute with random parameters ', time.time()-startExecutionTime)


##Get accuracy score
accuracyScore=accuracy_score(y_test,y_pred)
print('\n\n\n\n', 'accuracy score is : ',accuracyScore)

f1Score=f1_score(y_test,y_pred,average=None)
print('\n\n\n\n',' f1 score is : ',f1Score)

print('\nTime taken to execute with best parameters ', time.time()-startExecutionTime)


train_sizes, train_scores, test_scores = learning_curve(AdaBoostClassifier(n_estimators = 1,learning_rate=1,algorithm='SAMME' ),
                                                   X,
                                                   y, train_sizes = train_sizes, cv = 5,
                                                   scoring = 'accuracy',shuffle='True')
import numpy as np
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





