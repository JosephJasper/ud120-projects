#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
from pprint import pprint
from time import time

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [\
'poi',\
"bonus",\
"deferral_payments",\
"deferred_income",\
"director_fees",\
"exercised_stock_options",\
"expenses",\
"from_messages",\
"from_poi_to_this_person",\
'from_poi_rate',\
"from_this_person_to_poi",\
"long_term_incentive",\
"other",\
"restricted_stock",\
"restricted_stock_deferred",\
"salary",\
"shared_receipt_with_poi",\
"to_messages",\
'to_poi_rate',\
"total_payments",\
"total_stock_value"\
#
#"email_address",\
#Email address is too direct to POI's
#"loan_advances",\
#Only 3 uses and only POI's received
] 
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E')

#Examine the structure of the data and the ranges of the various features.
my_data_frame = pd.DataFrame.from_dict(data_dict,orient = 'index')

namelist = my_data_frame.index.tolist()
feature_list = my_data_frame.columns.get_values().tolist()

#Create modified features for poi contact rates
my_data_frame['to_poi_rate'] = 0.
my_data_frame['from_poi_rate'] = 0.

for a in namelist:
	if my_data_frame.at[a,'to_messages'] != 'NaN':
		my_data_frame.at[a,'from_poi_rate'] = \
		(1. * my_data_frame.at[a,'from_poi_to_this_person']) / \
		(1. * my_data_frame.at[a,'to_messages'])
	if my_data_frame.at[a,'from_messages'] != 'NaN':
		my_data_frame.at[a,'to_poi_rate'] = \
		(1. * my_data_frame.at[a,'from_this_person_to_poi']) / \
		(1. * my_data_frame.at[a,'from_messages'])

#Function that returns a dict as a description of a column in a dataframe.
#Meant to be used in feature_description() below.
def feature_describe(feature, dataframe = my_data_frame):
	feature_bag = {}
	if type(dataframe) == pd.core.frame.DataFrame:
		if type(feature) == str:
			if feature in dataframe.columns.get_values().tolist():
				feature_bag['frequency'] = \
				dataframe[feature].value_counts().to_dict()
				feature_values = dataframe[feature].value_counts().index
				if 'NaN' in feature_values:
					feature_values = feature_values.drop("NaN")
				feature_bag['max'] = feature_values.max()
				feature_bag['min'] = feature_values.min()
				feature_bag['NaN'] = 0
				if 'NaN' in feature_bag['frequency']:
					feature_bag['NaN'] = feature_bag['frequency']['NaN']
				feature_bag['uses'] = \
				sum(feature_bag['frequency'].values()) - \
				feature_bag['NaN']
			else:
				feature_bag['uses'] = 0
				feature_bag['NaN'] = 0
				feature_bag['frequency'] = {}
				feature_bag['min'] = 0.
				feature_bag['max'] = 0.
				print feature, "not present in dataframe."
		else:
			print "Argument error 1:"
			print "Please, input a string to describe in the first argument."
	else:
		print "Argument error 2:"
		print "Please, input a pandas DataFrame as the second argument."
	return feature_bag

#Function that returns a dictionary with a column (or columns) from a dataframe 
#as the key(s).
#The sub dictionarys describe various aspects of the column.
def feature_description(feature_s_, dataframe = my_data_frame):
	featurelist = {}
	if type(dataframe) == pd.core.frame.DataFrame:
		if type(feature_s_) == list:
			for a in feature_s_:
				featurelist[a] = feature_describe(a, dataframe)
		elif type(feature_s_) == str:
			featurelist[feature_s_] = feature_describe(feature_s_, dataframe)
		else:
			print "Argument error 1:"
			print "Please, input a string or a list to describe in the first "\
			"argument."
	else:
		print "Argument error 2:"
		print "Please, input a pandas DataFrame as the second argument."
	return featurelist

#pprint(feature_description(["to_poi_rate","from_poi_rate"]))

#Identify users with only "NaN" and a Poi value
for a in range(0,len(my_data_frame)):
	values = my_data_frame.iloc[a].value_counts()
	if  values['NaN'] > 19:
		print "Record with limited data:"
		print my_data_frame.iloc[a]

my_dataset = my_data_frame.to_dict(orient = "index")

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#http://scikit-learn.org/stable/modules/
#generated/sklearn.metrics.make_scorer.html
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer



###Quote and unquote sections based on which classifier used in the end.

#http://scikit-learn.org/stable/modules/
#model_evaluation.html#scoring-parameter
#3.3.1.4. Using multiple metric evaluation

precision = make_scorer(precision_score)
recall = make_scorer(recall_score)


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

def stratify(clf, ds = my_dataset, fl = feature_list, folds = 1000):
	data = featureFormat(ds, fl, sort_keys = True)
	labels, features = targetFeatureSplit(data)
	sss = StratifiedShuffleSplit(labels, folds,random_state=42)
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
	#print "F1:",f1_score(labels_test,pred)
	print "Precision:", precision_score(labels_tests,predictions)
	print "Recall:", recall_score(labels_tests,predictions)
	return predictions, labels_tests

fl1 = ["poi", "bonus", "salary", \
"from_poi_to_this_person", "from_this_person_to_poi", \
"from_messages", "to_messages"]

fl2 = ["poi", "bonus", "salary", \
"from_poi_rate", "to_poi_rate"]

features_full = [\
'poi',\
"bonus",\
"deferral_payments",\
"deferred_income",\
"director_fees",\
"exercised_stock_options",\
"expenses",\
"from_messages",\
"from_this_person_to_poi",\
"from_poi_to_this_person",\
"from_poi_rate",\
"long_term_incentive",\
"other",\
"restricted_stock",\
"restricted_stock_deferred",\
"salary",\
"shared_receipt_with_poi",\
"to_messages",\
"to_poi_rate",\
"total_payments",\
"total_stock_value"\
#
#"email_address",\
#Email address is too direct to POI's
#"loan_advances",\
#Only 3 uses and only POI's received
]

kn1 = KNeighborsClassifier(n_neighbors = 3, weights = "distance")

kn2 = KNeighborsClassifier(n_neighbors = 3, weights = "uniform")

minmax = MinMaxScaler(copy = False)

pl1 = Pipeline([("minmax",minmax),("kn1",kn1)])

feature_select = SelectKBest(k = 5)
SK_parameters = {\
"feature_select__k": (3,5,7)}

SVC1 = SVC(kernel = "linear")

pl2 = Pipeline([("minmax",minmax),("classifier",SVC1)])

pl3 = Pipeline([("minmax",minmax),("feature_select",feature_select),\
	('classifier',SVC1)])

gs1_param = {
	"classifier__C":[1,5,25],
	"classifier__kernel":["linear","rbf"],
	"feature_select__k": range(3,8)
}

gs1 = GridSearchCV(pl3,gs1_param)

gs1r = GridSearchCV(pl3,gs1_param, scoring = recall)

gs1p = GridSearchCV(pl3,gs1_param, scoring = precision)

gn1 = GaussianNB()

pl4 = Pipeline([("feature_select",feature_select),("classifier",gn1)])

gs2_param = {
	"feature_select__k": range(3,8)
}

gs2 = GridSearchCV(pl4,gs2_param)

gs2r = GridSearchCV(pl4,gs2_param, scoring = recall)

gs2p = GridSearchCV(pl4,gs2_param, scoring = precision)

pl5 = Pipeline([("feature_select",feature_select),("classifier",kn1)])

gs3_param = {
	"feature_select__k": range(3,8),
	"classifier__n_neighbors": [3,5,7,9],
	"classifier__weights" : ["uniform","distance"]
}

gs3 = GridSearchCV(pl5, gs3_param)

gs3r = GridSearchCV(pl5, gs3_param, scoring = recall)

gs3p = GridSearchCV(pl5, gs3_param, scoring = precision)

fl5b = [\
'poi',\
'total_stock_value',\
'exercised_stock_options',\
'salary',\
'bonus',\
'to_poi_rate'\
]

fl7b = [\
'poi',\
'total_stock_value',\
'exercised_stock_options',\
'salary',\
'bonus',\
'to_poi_rate',\
'restricted_stock',\
'expenses'\
]

kn3 = KNeighborsClassifier(n_neighbors = 3, weights = "uniform")

kn4 = KNeighborsClassifier(n_neighbors = 5, weights = "uniform")

kn5 = KNeighborsClassifier(n_neighbors = 7, weights = "uniform")

clfs = {
	"Old Email: Kneighbors 3 distance" : {
	"clf" : kn1,
	"feature list" : fl1
	},
	"New Email: Kneighbors 3 distance" : {
	"clf" : kn1,
	"feature list" : fl2 
	},
	"Old Email: Kneighbors 3 uniform" : {
	"clf" : kn2,
	"feature list" : fl1
	},
	"New Email: Kneighbors 3 uniform" : {
	"clf" : kn2,
	"feature list" : fl2
	},
	"Old Email: Minmax Kneighbors 3 distance" : {
	"clf" : pl1,
	"feature list" : fl1 
	},
	"New Email: Minmax Kneighbors 3 distance" : {
	"clf" : pl1,
	"feature list" : fl2
	},
	"Old Email: Minmax Linear SVC" : {
	"clf" : pl2,
	"feature list" : fl1
	},
	"New Email: Minmax Linear SVC" : {
	"clf" : pl2,
	"feature list" : fl2
	},
	"Full feature: Gridseach Minmax SelectK SVC" : {
	"clf" : gs1,
	"feature list" : features_full,
	"best parameters" : {},
	"best features" : [],
	"feature scores" : {},
	"best estimator" : {}
	},
	"Full feature: Gridseach Minmax SelectK SVC recall" : {
	"clf" : gs1r,
	"feature list" : features_full,
	"best parameters" : {},
	"best features" : [],
	"feature scores" : {},
	"best estimator" : {}
	},
	"Full feature: Gridseach Minmax SelectK SVC precision" : {
	"clf" : gs1p,
	"feature list" : features_full,
	"best parameters" : {},
	"best features" : [],
	"feature scores" : {},
	"best estimator" : {}
	},
	"Full feature: Gridseach SelectK GaussianNB" : {
	"clf" : gs2,
	"feature list" : features_full,
	"best parameters" : {},
	"best features" : [],
	"feature scores" : {},
	"best estimator" : {}
	},
	"Full feature: Gridseach SelectK GaussianNB recall" : {
	"clf" : gs2r,
	"feature list" : features_full,
	"best parameters" : {},
	"best features" : [],
	"feature scores" : {},
	"best estimator" : {}
	},
	"Full feature: Gridseach SelectK GaussianNB precision" : {
	"clf" : gs2p,
	"feature list" : features_full,
	"best parameters" : {},
	"best features" : [],
	"feature scores" : {},
	"best estimator" : {}
	},
	"Full feature: SelectK 5 GaussianNB" : {
	"clf" : pl4,
	"feature list" : features_full,
	"best features" : [],
	"feature scores" : {},
	},
	"Full feature: Gridseach SelectK Kneighbors" : {
	"clf" : gs3,
	"feature list" : features_full,
	"best parameters" : {},
	"best features" : [],
	"feature scores" : {},
	"best estimator" : {}
	},
	"Full feature: Gridseach SelectK Kneighbors recall" : {
	"clf" : gs3r,
	"feature list" : features_full,
	"best parameters" : {},
	"best features" : [],
	"feature scores" : {},
	"best estimator" : {}
	},
	"Full feature: Gridseach SelectK Kneighbors precision" : {
	"clf" : gs3p,
	"feature list" : features_full,
	"best parameters" : {},
	"best features" : [],
	"feature scores" : {},
	"best estimator" : {}
	},
	"Five best: Kneighbors 3 uniform" : {
	"clf" : kn3,
	"feature list" : fl5b
	},
	"Seven best: Kneighbors 3 uniform" : {
	"clf" : kn3,
	"feature list" : fl7b
	},
	"Five best: Kneighbors 5 uniform" : {
	"clf" : kn4,
	"feature list" : fl5b
	},
	"Seven best: Kneighbors 5 uniform" : {
	"clf" : kn4,
	"feature list" : fl7b
	},
	"Five best: Kneighbors 7 uniform" : {
	"clf" : kn5,
	"feature list" : fl5b
	},
	"Seven best: Kneighbors 5 uniform" : {
	"clf" : kn5,
	"feature list" : fl7b
	}
}

'''
	"" : {
	"clf" : 
	"feature list" : 
	"best parameters" : {}
	"best features" : []
	"feature scores" : {}
	"best estimator" : {}
	}
'''

def clf_tracker(clfs):
	if type(clfs) == dict:
		for a in clfs:
			print a
			t0 = time()
			predictions, labels_tests = stratify(clfs[a]['clf'],my_dataset,\
				clfs[a]['feature list'])
			clfs[a]['test-train seconds'] = round(time()-t0,3) 
			clfs[a]['precision'] = \
			round(precision_score(labels_tests, predictions),3)
			clfs[a]['recall'] = \
			round(recall_score(labels_tests, predictions),3)
			if 'best parameters' in clfs[a]:
				clfs[a]['best parameters'] = clfs[a]["clf"].best_params_
			if 'best features' in clfs[a]:
				try:
					selected_feature_values = \
					clfs[a]['clf'].best_estimator_.\
					named_steps["feature_select"].get_support(indices = True)
					for b in selected_feature_values:
						clfs[a]['best features'].\
						append(clfs[a]['feature list'][b+1])
				except:
					pass
				try:
					selected_feature_values = \
					clfs[a]['clf'].named_steps["feature_select"]\
					.get_support(indices = True)
					for b in selected_feature_values:
						clfs[a]['best features'].\
						append(clfs[a]['feature list'][b+1])
				except:
					pass
			if 'feature scores' in clfs[a]:
				try:
					feature_scores = \
					clfs[a]['clf'].best_estimator_.\
					named_steps["feature_select"].scores_
					for b in range(1,len(clfs[a]['feature list'])):
						feature_ph = clfs[a]['feature list'][b]
						clfs[a]['feature scores'][feature_ph] = \
						round(feature_scores[b - 1],3)
				except:
					pass
				try:
					feature_scores = \
					clfs[a]['clf'].named_steps["feature_select"].scores_
					for b in range(1,len(clfs[a]['feature list'])):
						feature_ph = clfs[a]['feature list'][b]
						clfs[a]['feature scores'][feature_ph] = \
						round(feature_scores[b - 1],3)
				except:
					pass
			if 'best estimator' in clfs[a]:
				try:
					clfs[a]['best estimator'] = \
					clfs[a]['clf'].best_estimator_
				except:
					pass
	else:
		print "Argument Error:"
		print "Please input a valid dictionary of classifiers to evaluate."
	return clfs

#Do not uncomment this section unless you have 2 hours to kill
#This code was created to generate the CLF_Evaluator that had the results 
#posted at the end of the project.
#To shorten this time remove the gridsearchcv classifiers 
#as they take the most time to run.
#clfs = clf_tracker(clfs)
#clf_dataframe = pd.DataFrame.from_dict(clfs,orient = 'index')
#clf_dataframe.to_csv("CLF_Evaluator.csv", sep = ",")



features_list = fl5b
clf = kn3

print "Final result:"
predictions, labels_tests = stratify(clf, my_dataset, features_list)

#Why not use the tester provided to test
#from tester import test_classifier

#test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
