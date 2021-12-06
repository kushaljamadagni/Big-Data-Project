#Importing some important libraries
import matplotlib.pyplot as plt
import numpy as np
import csv

f_obj = open("o.txt","r")#Enter the output file consisting of the metrics
#Storing the lines in s
s=f_obj.readlines()

#Converting the object into string
l=' '.join(str(e) for e in s)
#print(l)
metr=[]

m=l.split("[")
#print(len(m))
for i in range(1,len(m)):
 met=[]
 x=m[i].split("]")
 #print(x[0])
 
 y=x[0].split(",")
 #This is the value of Batch Number
 met.append(i)
 #This appends the accuracy values
 met.append(float(y[0]))
 #This appends the precision value
 met.append(float(y[1]))
 #This appends the recall values
 met.append(float(y[2]))
 #This appends the f1 score values
 met.append(float(y[3]))
 #This consists of all metrices for a particular batch
 metr.append(met)

#print(metr)

 


# field names
fields = ['Batch no','Accuracy', 'Precision', 'Recall', 'F1 Score']
	
# data rows of csv file
with open('sgd.csv', 'w') as f:
	
	# using csv.writer method from CSV package
	write = csv.writer(f)
	#Writes the column names first
	write.writerow(fields)
        #Then the data is written from the list
	write.writerows(metr)


