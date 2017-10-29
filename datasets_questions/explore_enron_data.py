#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print 'Number of entries in E+F data: ', len(enron_data)

#for a in enron_data:
#	print a

print 'Number of features of METTS, MARK: ', len(enron_data['METTS MARK'])

#for a in enron_data['METTS MARK']:
#	print a

POI = 0
NPOI = 0
for a in enron_data:
	if enron_data[a]['poi'] == 1:
		POI += 1
	else:
		NPOI += 1

print 'Number of POIs: ', POI
print 'Number of Non-POIs', NPOI

###Features in E + F dataset#################################################
#salary
#to_messages
#deferral_payments
#total_payments
#exercised_stock_options
#bonus
#restricted_stock
#shared_receipt_with_poi
#restricted_stock_deferred
#total_stock_value
#expenses
#loan_advances
#from_messages
#other
#from_this_person_to_poi
#poi
#director_fees
#deferred_income
#long_term_incentive
#email_address
#from_poi_to_this_person

###Names in E + F dataset####################################################
#METTS MARK
#BAXTER JOHN C
#ELLIOTT STEVEN
#CORDES WILLIAM R
#HANNON KEVIN P
#MORDAUNT KRISTINA M
#MEYER ROCKFORD G
#MCMAHON JEFFREY
#HORTON STANLEY C
#PIPER GREGORY F
#HUMPHREY GENE E
#UMANOFF ADAM S
#BLACHMAN JEREMY M
#SUNDE MARTIN
#GIBBS DANA R
#LOWRY CHARLES P
#COLWELL WESLEY
#MULLER MARK S
#JACKSON CHARLENE R
#WESTFAHL RICHARD K
#WALTERS GARETH W
#WALLS JR ROBERT H
#KITCHEN LOUISE
#CHAN RONNIE
#BELFER ROBERT
#SHANKMAN JEFFREY A
#WODRASKA JOHN
#BERGSIEKER RICHARD P
#URQUHART JOHN A
#BIBI PHILIPPE A
#RIEKER PAULA H
#WHALEY DAVID A
#BECK SALLY W
#HAUG DAVID L
#ECHOLS JOHN B
#MENDELSOHN JOHN
#HICKERSON GARY J
#CLINE KENNETH W
#LEWIS RICHARD
#HAYES ROBERT E
#MCCARTY DANNY J
#KOPPER MICHAEL J
#LEFF DANIEL P
#LAVORATO JOHN J
#BERBERIAN DAVID
#DETMERING TIMOTHY J
#WAKEHAM JOHN
#POWERS WILLIAM
#GOLD JOSEPH
#BANNANTINE JAMES M
#DUNCAN JOHN H
#SHAPIRO RICHARD S
#SHERRIFF JOHN R
#SHELBY REX
#LEMAISTRE CHARLES
#DEFFNER JOSEPH M
#KISHKILL JOSEPH G
#WHALLEY LAWRENCE G
#MCCONNELL MICHAEL S
#PIRO JIM
#DELAINEY DAVID W
#SULLIVAN-SHAKLOVITZ COLLEEN
#WROBEL BRUCE
#LINDHOLM TOD A
#MEYER JEROME J
#LAY KENNETH L
#BUTTS ROBERT H
#OLSON CINDY K
#MCDONALD REBECCA
#CUMBERLAND MICHAEL S
#GAHN ROBERT S
#MCCLELLAN GEORGE
#HERMANN ROBERT J
#SCRIMSHAW MATTHEW
#GATHMANN WILLIAM D
#HAEDICKE MARK E
#BOWEN JR RAYMOND M
#GILLIS JOHN
#FITZGERALD JAY L
#MORAN MICHAEL P
#REDMOND BRIAN L
#BAZELIDES PHILIP J
#BELDEN TIMOTHY N
#DURAN WILLIAM D
#THORN TERENCE H
#FASTOW ANDREW S
#FOY JOE
#CALGER CHRISTOPHER F
#RICE KENNETH D
#KAMINSKI WINCENTY J
#LOCKHART EUGENE E
#COX DAVID
#OVERDYKE JR JERE C
#PEREIRA PAULO V. FERRAZ
#STABLER FRANK
#SKILLING JEFFREY K
#BLAKE JR. NORMAN P
#SHERRICK JEFFREY B
#PRENTICE JAMES
#GRAY RODNEY
#PICKERING MARK R
#THE TRAVEL AGENCY IN THE PARK
#NOLES JAMES L
#KEAN STEVEN J
#TOTAL
#FOWLER PEGGY
#WASAFF GEORGE
#WHITE JR THOMAS E
#CHRISTODOULOU DIOMEDES
#ALLEN PHILLIP K
#SHARP VICTORIA T
#JAEDICKE ROBERT
#WINOKUR JR. HERBERT S
#BROWN MICHAEL
#BADUM JAMES P
#HUGHES JAMES A
#REYNOLDS LAWRENCE
#DIMICHELE RICHARD G
#BHATNAGAR SANJAY
#CARTER REBECCA C
#BUCHANAN HAROLD G
#YEAP SOON
#MURRAY JULIA H
#GARLAND C KEVIN
#DODSON KEITH
#YEAGER F SCOTT
#HIRKO JOSEPH
#DIETRICH JANET R
#DERRICK JR. JAMES V
#FREVERT MARK A
#PAI LOU L
#BAY FRANKLIN R
#HAYSLETT RODERICK J
#FUGH JOHN L
#FALLON JAMES B
#KOENIG MARK E
#SAVAGE FRANK
#IZZO LAWRENCE L
#TILNEY ELIZABETH A
#MARTIN AMANDA K
#BUY RICHARD B
#GRAMM WENDY L
#CAUSEY RICHARD A
#TAYLOR MITCHELL S
#DONAHUE JR JEFFREY M
#GLISAN JR BEN F

print "James Prentice total stock value: ", enron_data['PRENTICE JAMES']['total_stock_value']
print "Number of email messages from Wesley Colwell to POIs: ", enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print "Stock options exercised by Jeffrey K Skilling: ", enron_data['SKILLING JEFFREY K']['exercised_stock_options']
print "\n"
print "Kenneth Lay's total payments", enron_data['LAY KENNETH L']['total_payments']
print "Jeffrey Skilling's total payments", enron_data['SKILLING JEFFREY K']['total_payments']
print "Andrew Fastow's total payments", enron_data['FASTOW ANDREW S']['total_payments']

salary_Available = 0
email_Available = 0
no_Total_Payments = 0
poi_No_Total_Payments = 0
for a in enron_data:
	if enron_data[a]['email_address'] != 'NaN':
		email_Available += 1
	if enron_data[a]['salary'] != 'NaN':
		salary_Available += 1
	if enron_data[a]['total_payments'] == 'NaN':
		no_Total_Payments += 1
		if enron_data[a]['poi'] == 'NaN':
			poi_No_Total_Payments += 1

print "No. people with salary info: ", salary_Available
print "No. people with email address: ", email_Available
print "No. people without total payments information: ", no_Total_Payments, "\nAs a percent of whole E+F dataset: ", 100 * no_Total_Payments / len(enron_data), "%"
print "No. POI's without total payments information: ", poi_No_Total_Payments