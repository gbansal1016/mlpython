#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.pipeline import Pipeline

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_payments',  'total_stock_value', 'restricted_stock', 'long_term_incentive','expenses', 'from_poi_to_this_person'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#data_dict.pop("TOTAL",0)

my_dataset = data_dict
df = pd.DataFrame(my_dataset)
df = df.transpose()


## GB:: Data exploration
df = df.drop('email_address', axis=1)
df = df.drop('deferral_payments', axis=1)
df = df.drop('deferred_income', axis=1)
df = df.drop('director_fees', axis=1)
df = df.drop('loan_advances', axis=1)
df = df.drop('TOTAL')
df = df.drop('THE TRAVEL AGENCY IN THE PARK')

print('total number of poi', len(df.ix[df['poi']==1]))

#df = df.astype(float)
#print(df.columns.T) 
print(df.loc['SKILLING JEFFREY K'])

#print(df.head().describe())
#print(df.loc[:,'total_stock_value'])

#total_payments_col = (df.loc[:,'total_payments'].values).astype(float)
#print(total_payments_col.dtype)
#print(np.isnan(total_payments_col).any())

#GB Correlation analysis
#print('Correlation analysis')
#print(df.corr())

my_dataset = df.transpose().to_dict()

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, finance_features = targetFeatureSplit(data)


##GB-> Scale features
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(finance_features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# Trial 1. Naive Bayes Classifier
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

# Trial 2. Support Vector Machine
#from sklearn import svm
#Cvals = [1000.0]
#clf = svm.SVC(kernel='rbf', C=Cvals[0])

# Trial 3. Decision Tree Classifier

pipeline = Pipeline([
        ('imp', Imputer(missing_values='NaN', strategy='median')),
        ('selection', SelectKBest(k=5)),
        ('pca', PCA()),
        ('std', StandardScaler()),
        ('clf', tree.DecisionTreeClassifier(random_state = 53))
    ])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(pipeline, my_dataset, features_list)

from sklearn.metrics import accuracy_score, precision_score, recall_score


pipeline.fit(features_train, labels_train)

pred=pipeline.predict(features_test)

#print(np.where(pred>0))
idx = np.where(pred>0)


#print(np.array(labels_test).T)
print("Test data with POI(1)", np.where(np.array(labels_test)>=1))
print("Predicted values with POI(1)", np.where(pred>=1))
#print(pred.T)
accuracy = accuracy_score(labels_test, pred)
precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)
print("accuracy", accuracy)
print("precision", precision)
print("recall", recall)

