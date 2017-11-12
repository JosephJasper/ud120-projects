#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
predictions = clf.predict(features_test)
accuracy = accuracy_score(labels_test,predictions)
print "Accuracy of tree is", round(accuracy,2)

#for a in clf.feature_importances_:
#	print a

max_feature = max(clf.feature_importances_)
feature_length = len(clf.feature_importances_)
strongest_feature = 0
#for a in range(0,feature_length):
#	if clf.feature_importances_[a] == max_feature:
#		strongest_feature = a
#		print "Importance of most important feature:",round(max_feature,3)
#		print "Feature is ",strongest_feature,"th feature."

features = vectorizer.get_feature_names()
#print features[strongest_feature]

feature_threshold = .2
potential_signatures = []
for a in range(0,feature_length):
	line = []
	if clf.feature_importances_[a] > feature_threshold:
		line.append(a)
		line.append(clf.feature_importances_[a])
		line.append(features[a])
		potential_signatures.append(line)

if len(potential_signatures) > 0:
	print "The following should be evaluated for being a signature:"
	for a in potential_signatures:
		print a
else:
	print "No likely signatures found."

