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
import sys
import math
sys.path.append("../final_project/")
from  poi_email_addresses import poiEmails
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
#keys = enron_data.keys()
total=0
count=0
quantified_sal=0
total_pym_nan_cnt=0
for key, value in enron_data.items():
    total = total+1
    if (value["poi"]):
        count = count + 1
    if(not(math.isnan(float(value["salary"])))):
        quantified_sal = quantified_sal + 1
    if(math.isnan(float(value["total_payments"]))):
        total_pym_nan_cnt = total_pym_nan_cnt + 1
            
print("total", total)
print("Number of POI: %d" , count)

print("Nan payment counts ids %d", total_pym_nan_cnt)
print("Quantified salary %d", quantified_sal)
      
print("Email addresses of all POI %d",len(poiEmails()))          

text_file = open("../final_project/poi_names.txt", "r")
lines = text_file.readlines()
count=0
for line in lines:
    if((len(line) > 1) and line[1]=='y'):
        count = count + 1
        
print("Number of POI: %d" , count)      

print("Stock value for Mr. James Prentice %f", enron_data["PRENTICE JAMES"]["total_stock_value"])
print("Stock value for Mr. Wesley Colwell %d", enron_data["Colwell Wesley".upper()]["total_stock_value"])
print("Stock options exercised by Mr. Skilling %d", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

print("Total Payments to Mr. Skilling %d", enron_data["SKILLING JEFFREY K"]["total_payments"])
print("Total Payments to  Mr. Lay %d", enron_data["LAY KENNETH L"]["total_payments"])
print("Total Payments to  Mr. Fastow %d", enron_data["FASTOW ANDREW S"]["total_payments"])

text_file.close()



#count[1 for d in enron_data if d.get('poi')==1]