import csv
import difflib
from sklearn import svm
from scipy import sparse
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
import time
import nltk
import numpy as np
from nltk.tokenize import wordpunct_tokenize
np.set_printoptions(suppress =True, precision=3)
import copy

#v = open('sklearn_prediction_new.csv')
#out1 = open('sklearn_prediction_refined.csv', 'w', encoding="utf8")
#out = csv.writer(out1)
v1 = open('test.csv',encoding="utf8")
#r = csv.reader(v)
test_file = csv.reader(v1)
senti_labels = []
threshold = 0.5
senti_labels = ["I can't tell", 'Negative', 'Neutral / author is just sharing information', 'Positive', 'Tweet not related to weather condition']
when_labels = ['current (same day) weather', 'future (forecast)', "I can't tell", 'past weather']
weather_labels = ['clouds', 'cold', 'dry', 'hot', 'humid', 'hurricane', "I can't tell", 'ice', 'other', 'rain', 'snow', 'storms', 'sun', 'tornado', 'wind']

classify = joblib.load('joblib_500.pkl')
tdf = classify[0]
lsa = classify[1]
norm = classify[2]
clf = classify[3]
def cluster_helper(state):
    flag = False
    #print("State -----> \n\n\n\n", state)
    clustered_data[state] = []
    v1.seek(0)
    for row in csv.reader(v1):
        if(flag == True):
            location_set = row[2].split(',') + row[3].split(',')
            matching_tweet = difflib.get_close_matches(state, location_set)
            if(len(matching_tweet) != 0):
                clustered_data[state].append(row[1])
        flag = True

def cluster_tweets(states):
    for state in states:
        cluster_helper(state)
        
full_cluster = {}
final_prediction = {}
output_labels = {}

def cluster_data_daywise_helper(state):
    if(len(clustered_data[state]) <= 0):
        full_cluster[state] = {}
        print("NO MATCHING TWEET FOR THIS STATE")
        return
    test_data = tdf.transform(clustered_data[state])
    test_data = lsa.transform(test_data)
    test_data = norm.transform(test_data, copy= False)
    #print("test_data shape is -->", test_data.shape)
    prediction = np.array(clf.predict(test_data))
    prediction = np.abs(prediction*(prediction > 0))
    prediction[prediction > 1] = 1
    prediction[prediction < 0] = 0
    full_cluster[state] = {}
    
    final_prediction[state] = {}
    output_labels[state] = {}
    for l in when_labels:
        full_cluster[state][l] = []
        final_prediction[state][l] = []
        output_labels[state][l] = []
    tn = 0    
    for prow in prediction:
        when = prow[5:9].tolist()
        full_cluster[state][when_labels[when.index(max(when))]].append(prow)#clustered_data[state][tn]) # if you want prediction
        tn += 1
    
def cluster_data_daywise(states):
    for state in states:
        cluster_data_daywise_helper(state)
        
def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None        



def predict_weather_helper(state):
    flag = True
    if(len(full_cluster[state]) <= 0):
        return
    for i in full_cluster[state]:  #for each label
        if(len(full_cluster[state][i]) > 0):
            myInt = len(full_cluster[state][i])
            temp = [sum(x) for x in zip(*(full_cluster[state][i]))]
            newList = [x / myInt for x in temp]
            final_prediction[state][i] = newList

    for row in final_prediction[state]:
        new_row = []
        weather_collection = []
        if(len(final_prediction[state][row]) <= 0):
            continue
        senti = final_prediction[state][row][0:5]
            
        predict_senti = senti_labels[senti.index(max(senti))]
        
        weather = final_prediction[state][row][9:24]            
        #print("SSSSS ---> ",weather)
        weather_collection.append(weather_labels[weather.index(max(weather))])
        weather_collection.append(weather_labels[weather.index(second_largest(weather))])
        new_row.append(predict_senti)
        if(weather_collection[0] == "I can't tell"):
            output_labels[state][row].append(weather_collection[1])
        else:
            output_labels[state][row].append(weather_collection[0])
            #new_row.append(weather_collection[0])
        #output_labels[state][row].append(new_row)

def predict_weather(states):
    for state in states:
        predict_weather_helper(state)
            
    
states = ['california', 'illionis', 'newyork', 'texas']
clustered_data = {}
cluster_tweets(states)
cluster_data_daywise(states)
predict_weather(states)

#print("FINAL",final_prediction)
#print(final_prediction)

print("TWEETS FROM TEXAS TELLS:")
print("PAST DAY:  ",output_labels['texas']["past weather"])
print("CURRENT DAY:  ",output_labels['texas']["current (same day) weather"])
print("FUTURE DAY:  ",output_labels['texas']["future (forecast)"])
print("\n")
print("TWEETS FROM ILLIONIS TELLS:")
print("PAST DAY:  ",output_labels['illionis']["past weather"])
print("CURRENT DAY:  ",output_labels['illionis']["current (same day) weather"])
print("FUTURE DAY:  ",output_labels['illionis']["future (forecast)"])
print("\n")
print("TWEETS FROM NEW YORK TELLS:")
print("PAST DAY:  ",output_labels['newyork']["past weather"])
print("CURRENT DAY:  ",output_labels['newyork']["current (same day) weather"])
print("CUTTENT DAY:  ",output_labels['newyork']["future (forecast)"])
print("\n")
print("TWEETS FROM CALIFORNIA TELLS:")
print("PAST DAY:  ",output_labels['california']["past weather"])
print("CURRENT DAY:  ",output_labels['california']["current (same day) weather"])
print("FUTURE DAY:  ",output_labels['california']["future (forecast)"])
