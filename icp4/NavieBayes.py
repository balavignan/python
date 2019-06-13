import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd

# sklearn.naive_bayes.GaussianNB(priors=None, var_smoothing=1e-09)

# Importing the dataset

dataset = pd.read_csv('D:\Python\icp4\glass.csv')


# looking at the first 9 values of the dataset

dataset.head()


# Spliting the dataset in independent and dependent variables

X = dataset.iloc[:, :9].values
Y = dataset['Type'].values


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.40)


# Fitting Naive Bayes Classification to the Training set with linear kernel

nvclassifier = GaussianNB()
nvclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = nvclassifier.predict(X_test)
# lets see the actual and predicted value side by side
y_compare = np.vstack((y_test, y_pred)).T
# actual value on the left side and predicted value on the right hand side
# printing the top 9 values
#print(y_compare[:3, :])


# Accuracy of NavesBayes on Training Sets
print('Accuracy of Naive Bayes classifier on training set: {:.2f}'
     .format(nvclassifier.score(X_train, y_train)))

# Accuracy of NavesBayes on Testing Sets
print('Accuracy of Naive Bayes classifier on test set: {:.2f}'
     .format(nvclassifier.score(X_test, y_test)))

# print the accuracy score of predicted and actual values on test set
print('\n\nAccuracy of the Naive Bayes Classification on Test Data Prediction is : ', metrics.accuracy_score(y_test, y_pred))


# Fitting SVC Classification to the Training set with linear kernel
svc_linear = SVC(kernel='linear', C=1, gamma=0.1).fit(X_train, y_train)
# Accuracy of SVM Linear kernel on Training set
print('\n\nAccuracy of the SVM Linear Kernel Classification on training part is: ', svc_linear.score(X_train, y_train))
# Accuracy of SVM Linear Kernel on Test Set
print('\n\nAccuracy of the SVM Linear Kernel Classification on testing part is: ', svc_linear.score(X_test, y_test))


# Fitting SVC Classification to the Training set with rbf  kernel
#svc_rbf = SVC(kernel='rbf', C=1, gamma=0.1).fit(X_train, y_train)
# Accuracy of SVM RBF kernel on Training set
#print('\n\nAccuracy of the SVM RBF Kernel Classification on training part is: ', svc_rbf.score(X_train, y_train))
# Accuracy of SVM RBF Kernel on Test Set
#print('\n\nAccuracy of the SVM  RBF Kernel Classification on testing part is: ', svc_rbf.score(X_test, y_test))

svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_test, y_test) * 100, 2)
print("\n\nAccuracy of the SVM  RBF Kernel Classification on testing part is:", acc_svc)

acc_svc = round(svc.score(X_train, y_train) * 100, 2)
print("\n\nAccuracy of the SVM  RBF Kernel Classification on train part is:", acc_svc)
#print("prediction",Y_pred)