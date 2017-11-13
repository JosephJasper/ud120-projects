#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
pre_dataset = data_dict
pre_dataset.pop('TOTAL')

#Examine the structure of the data and the ranges of the various features.
'''
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

print "Features"
for a in featurelist:
	print a,", Unique Values: " ,len(featurelist[a]["Values"]),\
	", Minimum Value:",min(featurelist[a]["Values"]),",Maximum Value:",\
	max(featurelist[a]["Values"]), featurelist[a]

'''
#Identify key financial measures, scale and then use PCA to find 1 measure
'''
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
'''
#Originally tried to leverage all financial features in hopes that each 
#component could help inform a PCA
#financial_PCA_features = ["salary","bonus","long_term_incentive","deferred_income",\
#"deferral_payments","other","expenses","director_fees",\
#"exercised_stock_options","restricted_stock","restricted_stock_deferred"]

#Utilized summary features instead
'''
financial_PCA_features = ["total_payments","total_stock_value"]

financial_PCA_Data = []
for a in pre_dataset:
	row_data = []
	for b in financial_PCA_features:
		if pre_dataset[a][b] == "NaN":
			row_data.append(0)
		else:
			row_data.append(pre_dataset[a][b])
	financial_PCA_Data.append(row_data)

financial_Scaler = MinMaxScaler()
print financial_PCA_Data
financial_PCA_Data = financial_Scaler.fit_transform(financial_PCA_Data)
print financial_PCA_Data
financial_PCA = PCA(n_components = 1)
financial_PCA_Data = financial_PCA.fit_transform(financial_PCA_Data)
print financial_PCA_Data
print len(financial_PCA_Data)

print financial_PCA.explained_variance_ratio_
'''
#for a in pre_dataset:
#	if 'total_payments' not in pre_dataset[a] or \
#	pre_dataset[a]['total_payments'] == "NaN":
#		pre_dataset[a]['total_payments'] = 0
#	if 'total_stock_value' not in pre_dataset[a] or \
#	pre_dataset[a]['total_stock_value'] == "NaN":
#		pre_dataset[a]['total_stock_value'] = 0
#	pre_dataset[a]['total_compensation'] = pre_dataset[a]['total_payments'] + \
#	pre_dataset[a]['total_stock_value']
#	if pre_dataset[a]['total_compensation'] != 0:
#		pre_dataset[a]['stock_compensation_rate'] = \
#	(pre_dataset[a]['total_stock_value'] * 1.) / \
#	(pre_dataset[a]['total_compensation'] * 1.)
#
#for a in pre_dataset:
#	if pre_dataset[a]['to_messages'] != "NaN" and \
#	pre_dataset[a]['from_messages'] != "NaN":
#		pre_dataset[a]['poi_email_rate'] = (1. * \
#		(pre_dataset[a]['from_poi_to_this_person'] + \
#		(pre_dataset[a]['from_this_person_to_poi']))) / \
#		(1. * (pre_dataset[a]['from_messages'] + \
#		(pre_dataset[a]['to_messages'])))



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
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=100, test_size=.3,random_state=42)


for train_index, test_index in sss.split(features, labels):
#	features_train, features_test = features(train_index), features(test_index)
#	labels_train, labels_test = labels(train_index), labels(test_index)
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

clf = clf.fit(features_train,labels_train)
pred = clf.predict(features_test)




#Why not use the tester provided to test
from tester import test_classifier

test_classifier(clf, my_dataset, features_list)

#from sklearn.metrics import f1_score
#print f1_score(labels_test,pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.



dump_classifier_and_data(clf, my_dataset, features_list)



#Forum help
#https://discussions.udacity.com/t/invalid-parameter-c-for-estimator-pipeline/42100/2
#https://discussions.udacity.com/t/what-is-pipeline/185160/3