#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###


def classify(features_train,labels_train,c_Value):
	clf = SVC(kernel = "rbf", C = c_Value)
	clf.fit(features_train,labels_train)
	return clf


def evaluate_C(c_Value):
	t0 = time()
	clf = classify(features_train,labels_train,c_Value)
	print "Training time:", round(time()-t0,3), "s"
	t1 = time()
	pred = clf.predict(features_test)
	print "Predicting time:", round(time()-t1,3), "s"
	accuracy = accuracy_score(pred, labels_test)
	print "Accuracy: ", accuracy
	print "10th prediction: ",pred[10]
	print "26th prediction: ",pred[26]
	print "50th prediction: ",pred[50]
	chris = sum(pred)
	sara = len(pred) - sum(pred)
	print "Total emails evaluated: ",len(pred)
	print "no. of emails attributed to Chris: ",chris
	print "no. of emails attributed to Sara: ",sara


###features_train = features_train[:len(features_train)/100] 
###labels_train = labels_train[:len(labels_train)/100] 

###for i in (10.0,100.0,1000.0,10000.0):
###	print "------------------------------------------------"
###	print "C is ", i
###	evaluate_C(i)

evaluate_C(10000.0)

#########################################################