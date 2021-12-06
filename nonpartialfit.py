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
def logreg(df):
 data=np.array(df.select("features").collect())#this selects the features column from the dataframe 
 labeled=np.array(df.select("label").collect())#this selects the labels from the dataframe
 nosamp,nox,noy=data.shape #finds the total no of samples the no of x and no of y
 data = data.reshape((nosamp,nox*noy)) #reshape the data we created from the features 
 trainforx,testforx,trainfory,testfory=train_test_split(data,labeled,test_size=0.4,random_state=42)#split the dataset into seperate training and testing for x and y
 logreg = LogisticRegression()#call the logistic regression function
 logreg.fit(trainforx,trainfory)#fit the x and y parameters of training using the logistic regression we got
 y_pred=logreg.predict(testforx)#predict the y component from the test data
 print('Accuracy score: {}'.format(accuracy_score(testfory,y_pred)))#print the accuracy score
 print('Precision score: {}'.format(precision_score(testfory,y_pred)))#print the precission score
 print('Recall score: {}'.format(recall_score(testfory,y_pred)))#print the recall score 
 print('F1 score: {}'.format(f1_score(testfory,y_pred)))#print the f1 score.
