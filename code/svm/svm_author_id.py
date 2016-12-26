#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("..\\tools\\")

from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
Cvals = [10000.0]

for c in Cvals:
    print("Iterating for c",c)
    features_train, features_test, labels_train, labels_test = preprocess()
    t0 = time()
    clf = svm.SVC(kernel='rbf', C=c)
    
    #features_train = features_train[:len(features_train)//100] 
    #labels_train = labels_train[:len(labels_train)//100] 
                                
    clf.fit(features_train,labels_train)
    print ("training time:", round(time()-t0, 3), "s")
    
    
    t0 = time()
    pred=clf.predict(features_test)
    print ("prediction time:", round(time()-t0, 3), "s")
    
    accuracy = accuracy_score(labels_test, pred)
    print(accuracy)
    
    
    #print("class for 10",pred[10])
    #print("class for 26",pred[26])
    #print("class for 50",pred[50])
    print("Emails by Chris", sum(pred))
#########################################################
### your code goes here ###

#########################################################


