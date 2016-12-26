# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 16:06:36 2016

@author: gbans6
"""

def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    
    ### create classifier#TODO
    clf = GaussianNB()
    clf.fit(features_train, labels_train)

    ### fit the classifier on the training features and labels
    #TODO

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = accuracy_score(labels_test, pred)
    
    
    return accuracy