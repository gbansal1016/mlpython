#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import numpy as np
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
### it's all yours from here forward!  
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
print("full set", len(features))
print("training set", len(features_train))
print("test set", len(features_test))
clf.fit(features_train, labels_train)

pred=clf.predict(features_test)

#print(np.where(pred>0))
idx = np.where(pred>0)

#print(np.array(labels_test)[idx])
print(pred.T)
print(np.array(labels_test).T)
accuracy = accuracy_score(labels_test, pred)
precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)
print("accuracy", accuracy)
print("precision", precision)
print("recall", recall)
