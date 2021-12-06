import json
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.sql.functions import length
from pyspark.ml import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
sparkc=SparkContext("local[2]", "BigDataproject")
sparkstreamc=StreamingContext(sparkc,1)
spark = SparkSession.builder.getOrCreate()
#function used to flatten the data recieved into an rdd
def rdd_to_json(x):
    flat_json_li=json.loads(x).values()#loads the json file recieved
    for dicts in flat_json_li:
        for key in dicts:
            dicts[key]=str(dicts[key])
    return(flat_json_li)#traverse through. it and return as a dictionary
#processing
def cleaning(df):
	df=df.withColumn('length',length(df['feature1'])) #creates a dataframe with the features column
	token=Tokenizer(inputCol="feature1", outputCol="tokenized")#tokenizes the values from the feature column
	stoprem=StopWordsRemover(inputCol='tokenized',outputCol='stopwords')#removes the stopwords from the tokenized values
	count_vec=CountVectorizer(inputCol='stopwords',outputCol='count_vec')#this is used to convert the stopwords into a countvector
	idf = IDF(inputCol="count_vec", outputCol="tfandidf")#it is used to find the frequency of each word
	to_num=StringIndexer(inputCol='feature2',outputCol='label')#this basically converts each string to an integer value 
	clean=VectorAssembler(inputCols=['tfandidf','length'],outputCol='features')#this basically gives the final output of features
	data_pipe=Pipeline(stages=[to_num,token,stoprem,count_vec,idf,clean])#this is a pipeline to perform the above functionalities
	cleaner=data_pipe.fit(df)#this basically fits in the cleaned data
	final_df_cleaned=cleaner.transform(df)
	#print('preprocess done')
	#final_df_cleaned.show()
	#print('preprocess done')
	#clusterin(final_df_cleaned)#call to  the clustering  model
	#randomfor(final_df_cleaned)#call to the random forest model
	#logreg(final_df_cleaned)#call to the logisitic regression model
	sgdclass(final_df_cleaned)#call to the  standardgradientdescent model
#models
#logistic regression model
def logreg(df):
 data=np.array(df.select("features").collect())#this selects the features column from the dataframe 
 labeled=np.array(df.select("label").collect())#this selects the labels from the dataframe
 nosamp,nox,noy=data.shape #finds the total no of samples the no of x and no of y
 data = data.reshape((nosamp,nox*noy)) #reshape the data we created from the features 
 trainforx,testforx,trainfory,testfory=train_test_split(data,labeled,test_size=0.4,random_state=42)#split the dataset into seperate training and testing for x and y
 logreg = LogisticRegression()#call the logistic regression function
 logreg.fit(trainforx,trainfory)#fit the x and y parameters of training using the logistic regression we got
 y_pred=logreg.predict(testforx)#predict the y component from the test data
 #print('Accuracy score: {}'.format(accuracy_score(testfory,y_pred)))#print the accuracy score
 #print('Precision score: {}'.format(precision_score(testfory,y_pred)))#print the precission score
 #print('Recall score: {}'.format(recall_score(testfory,y_pred)))#print the recall score 
 #print('F1 score: {}'.format(f1_score(testfory,y_pred)))#print the f1 score.
 metrics=[accuracy_score(testfory,y_pred),precision_score(testfory,y_pred),recall_score(testfory,y_pred),f1_score(testfory,y_pred)] #making a list consisting of evaluation metrics 
 print(metrics) #printing the same list for better understanding instead of printing four different lines 

	
def sgdclass(df):
 data=np.array(df.select("features").collect())
 labeled=np.array(df.select("label").collect())
 nosamp,nox,noy=data.shape
 data = data.reshape((nosamp,nox*noy))
 trainforx,testforx,trainfory,testfory=train_test_split(data,labeled,test_size=0.4,random_state=42)
 #clf=RandomForestClassifier(n_estimators=100)
 #clf.fit(X_train,y_train)
 #y_pred=clf.predict(X_test)
 clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000,tol=1e-4))
 clf.partial_fit(trainforx,trainfory)
 y_pred=clf.predict(testforx)
 #print('Accuracy score: {}'.format(accuracy_score(testfory,y_pred)))
 #print('Precision score: {}'.format(precision_score(testfory,y_pred)))
 #print('Recall score: {}'.format(recall_score(testfory,y_pred)))
 #print('F1 score: {}'.format(f1_score(testfory,y_pred)))
 metrics=[accuracy_score(testfory,y_pred),precision_score(testfory,y_pred),recall_score(testfory,y_pred),f1_score(testfory,y_pred)]
 print(metrics)
#model for random forest
def randomfor(df):
 data=np.array(df.select("features").collect())    # there were two 
 labeled=np.array(df.select("label").collect())
 nosamp,nox,noy=data.shape
 data=data.reshape((nosamp,nox*noy))
 trainforx,testforx,trainfory,testfory=train_test_split(data,labeled,test_size=0.33, random_state=42)
 clf=RandomForestClassifier(n_estimators=100) #100 decision trees 
 clf.partial_fit(trainforx,trainfory)   #partially fittinng the data  
 y_pred=clf.predict(testforx)     # predicting the output for test data 
 #print('Accuracy score: {}'.format(accuracy_score(testfory,y_pred)))
 #print('Precision score: {}'.format(precision_score(testfory,y_pred)))
 #print('Recall score: {}'.format(recall_score(testfory,y_pred)))
 #print('F1 score: {}'.format(f1_score(testfory,y_pred)))
 metrics=[accuracy_score(testfory,y_pred),precision_score(testfory,y_pred),recall_score(testfory,y_pred),f1_score(testfory,y_pred)]  # printing all the metrics in single list for ease in taking inputs for plots and to maintain uniformity
 print(metrics)
	
def RDDtoDf(time,rdd): #function to convert the rdd which was streamed into a dataframe
    print(f"========= {str(time)} =========") #this will give us the time at which the batch file has been delivered to spark
    try:
        if(rdd==[] or rdd is None or rdd==[[]]):#checks if rdd is empty or not
            return 
        rdd=rdd.flatMap(lambda x:rdd_to_json(x))#flatmaps the rdd and calls the rdd_to_json function to convert to json
        df=spark_context.createDataFrame(rdd,["feature0","feature1","feature2"]) #creates a dataframe from the rdd which is flatmapped
        cleaning(df)#calls the cleaning function to preprocess and clean data
    except:
        print("No Data") #if no data has been read the exception will be called

rdd= sparkstreamc.socketTextStream("localhost", 6100) # creates an input from TCP source hostname and port. Data recieved is UTF-8 encoded
rdd.foreachRDD(RDDtoDf) #calls the function for each RDD in Dstream 
sparkstreamc.start()
sparkstreamc.awaitTermination()
sparkstreamc.stop()  
  

def clusterin(df):
 #Extracting the x variables i.e the features
 data=np.array(df.select("features").collect())
 #The Y variables or the target variables
 labeled=np.array(df.select("label").collect())
 nosamp,nox,noy=data.shape
 data=data.reshape((nosamp,nox*noy))
 #Splitting the data into training set for x and y and testing set for x and y
 trainforx,testforx,trainfory,testfory=train_test_split(data,labeled,test_size=0.4,random_state=42)
 clustering_algo = MiniBatchKMeans(n_clusters=2, random_state=12)
 #Epoch is the number of iterations for the centroid calculations
 epochs = 35

 for k in range(epochs):
  
  X_batch, Y_batch = trainforx, trainfory
  clustering_algo.partial_fit(X_batch, Y_batch)
 predY_train = []
 
 predY = clustering_algo.predict(testforx)
 predY_train.extend(predY.tolist())
 #print('Accuracy score: {}'.format(accuracy_score(testfory,predY_train)))
 #print('Precision score: {}'.format(precision_score(testfory,predY_train)))
 #print('Recall score: {}'.format(recall_score(testfory,predY_train)))
 #print('F1 score: {}'.format(f1_score(testfory,predY_train)))
 #cluster_centers = clustering_algo_predict.cluster_centers_
 
 #print(clustering_algo.cluster_centers_)
 #print(predY_train)
 metrics=[accuracy_score(testfory,predY_train),precision_score(testfory,predY_train),recall_score(testfory,predY_train),f1_score(testfory,predY_train)] 
 print(metrics)

	
