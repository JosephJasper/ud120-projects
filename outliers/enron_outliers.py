#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL")
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below


for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

print "Highest Salary and Bonus"
for a in data_dict:
	if data_dict[a]['salary'] != "NaN" and data_dict[a]['bonus'] != "NaN":
		if data_dict[a]['salary'] > 1000000 and data_dict[a]['bonus'] > 5000000:
			print "Employee:", a, " Salary:", data_dict[a]['salary'], " Bonus:", data_dict[a]['bonus']
