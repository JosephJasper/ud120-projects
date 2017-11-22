#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [\
'poi',\
#"to_messages",\
#Likely not a useful feature in it's own right
"bonus",\
"deferral_payments",\
"deferred_income",\
"director_fees",\
"exercised_stock_options",\
"expenses",\
'from_poi_rate',\
"long_term_incentive",\
"other",\
"restricted_stock",\
"restricted_stock_deferred",\
"salary",\
"shared_receipt_with_poi",\
"total_payments",\
'to_poi_rate',\
"total_stock_value"\
#"email_address",\
#Email address is too direct to POI's
#"from_poi_to_this_person",\
#May skew effectiveness in from_poi_rate
#"loan_advances",\
#Only 3 uses and only POI's received
#"from_messages",\
#Likely not a useful feature in it's own right
#"from_this_person_to_poi",\
#May skew effectiveness in to_poi_rate
] 
# You will need to use more features



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
pre_dataset = data_dict
pre_dataset.pop('TOTAL')

#Examine the structure of the data and the ranges of the various features.
for a in pre_dataset:
	if pre_dataset[a]["from_messages"] == "NaN":
		pre_dataset[a]["to_poi_rate"] = 0.
	if pre_dataset[a]["from_messages"] != "NaN":
		pre_dataset[a]["to_poi_rate"] = \
		(1. * pre_dataset[a]['from_this_person_to_poi']) / \
		(1. * pre_dataset[a]['from_messages'])
	if pre_dataset[a]['to_messages'] == "NaN":
		pre_dataset[a]["from_poi_rate"] = 0.
	if pre_dataset[a]['to_messages'] != "NaN":
		pre_dataset[a]["from_poi_rate"] = \
		(1. * pre_dataset[a]['from_poi_to_this_person']) / \
		(1. * pre_dataset[a]['to_messages'])



namelist = []
featurelist = {}
for a in pre_dataset:
	if a not in namelist:
		namelist.append(a)
	for b in pre_dataset[a]:
		if b not in featurelist:
			featurelist[b] = {"NaN": 0,"Uses": 0, "Values": []}
		if b in featurelist:
			if pre_dataset[a][b] == "NaN":
				featurelist[b]["NaN"] += 1
			else:
				featurelist[b]["Uses"] += 1
				if pre_dataset[a][b] not in featurelist[b]["Values"]:
					featurelist[b]["Values"].append(pre_dataset[a][b])

print "Features available:"
for a in featurelist:
	print a,", Unique Values: " ,len(featurelist[a]["Values"]),\
	", Minimum Value:",min(featurelist[a]["Values"]),",Maximum Value:",\
	max(featurelist[a]["Values"]), featurelist[a]
print "\n"

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
#from sklearn.model_selection import GridSearchCV
#http://scikit-learn.org/stable/modules/
#generated/sklearn.metrics.make_scorer.html
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer

feature_select = SelectKBest(k = 5)

#minmax = MinMaxScaler(copy = False)


###Quote and unquote sections based on which classifier used in the end.
'''
classifier = KNeighborsClassifier()
classifier_parameters = {\
"classifier__weights":["uniform","distance"],\
#"classifier__algorithm":['auto','ball_tree','kd_tree','brute'],\
#"classifier__leaf_size":[5,15,30],\
"classifier__n_neighbors":(3,5)\
}
'''

classifier = GaussianNB()
classifier_parameters = {}

'''
classifier = SVC(random_state = 42)
classifier_parameters = {\
"classifier__C":[1,5,25],\
"classifier__kernel":["linear","rbf","poly"]\
}
'''

clf = Pipeline([\
#	('minmax',minmax),\
	('feature_select',feature_select),\
	('classifier',classifier)])

'''
parameters = {\
"feature_select__k": (3,5,7)}

if len(classifier_parameters) > 0:
	for a in classifier_parameters:
		parameters[a] = classifier_parameters[a]
'''

#http://scikit-learn.org/stable/modules/
#model_evaluation.html#scoring-parameter
#3.3.1.4. Using multiple metric evaluation

#precision = make_scorer(precision_score)
#recall = make_scorer(recall_score)

#clf = GridSearchCV(pipeline, parameters, scoring = 'f1')


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/
###sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(labels, 1000, test_size=.20,random_state=42)

predictions = []
labels_tests = []

for train_index, test_index in sss:
	features_train = []
	features_test  = []
	labels_train   = []
	labels_test    = []
	for ii in train_index:
		features_train.append( features[ii] )
		labels_train.append( labels[ii] )
	for jj in test_index:
		features_test.append( features[jj] )
		labels_test.append( labels[jj] )
	clf.fit(features_train,labels_train)
	pred = clf.predict(features_test)
	labels_tests.extend(labels_test) 
	predictions.extend(pred)

#https://stackoverflow.com/questions/39839112/
#the-easiest-way-for-getting-feature-names-after-running-selectkbest-in-scikit-
#le
selected_feature_values = feature_select.get_support(indices = True)
print "Features selected:"
for a in selected_feature_values:
	print features_list[a+1]
print '\n'

print "Total POI's:",sum(labels)
print "Total dataset:",len(labels)
print '\n'

print "F1:",f1_score(labels_test,pred)
print "Precision:", precision_score(labels_tests,predictions)
print "Recall:", recall_score(labels_tests,predictions)

#Why not use the tester provided to test
#from tester import test_classifier

#test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

#Forum help
#https://discussions.udacity.com/t/invalid-parameter-c-for-estimator-pipeline/42100/2
#https://discussions.udacity.com/t/what-is-pipeline/185160/3
#https://discussions.udacity.com/t/selectkbest-and-adding-removing-features/162469/4