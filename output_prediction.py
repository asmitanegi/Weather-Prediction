import csv
v = open('sklearn_prediction_new.csv')
out1 = open('sklearn_prediction_refined.csv', 'w',encoding="utf8")
out = csv.writer(out1)
v1 = open('test_small.csv',encoding="utf8")
r = csv.reader(v)
test_file = csv.reader(v1)
senti_labels = []
threshold = 0.5

senti_labels = ["I can't tell", 'Negative', 'Neutral / author is just sharing information', 'Positive', 'Tweet not related to weather condition']
when_labels = ['current (same day) weather', 'future (forecast)', "I can't tell", 'past weather']
weather_labels = ['clouds', 'cold', 'dry', 'hot', 'humid', 'hurricane', "I can't tell", 'ice', 'other', 'rain', 'snow', 'storms', 'sun', 'tornado', 'wind']

flag = False
test_row = []
for row in csv.reader(v1):
    if(flag == True):
        test_row.append(row[1])
    flag = True
itr = 0
for row in csv.reader(v):
    new_row = []
    weather_collection = []
    new_row.append(test_row[itr])
    print("TWEET IS: ",test_row[itr])
    itr += 1
    row = [ float(x) for x in row ]
    senti = row[1:6]
    predict_senti = senti_labels[senti.index(max(senti))]
    print("Sentiment prediction: ",predict_senti)
    when = row[6:10]
    predict_when = when_labels[when.index(max(when))]
    print("When prediction: ", predict_when)
    weather = row[10:24]
    predict_weather = weather_labels[weather.index(max(weather))]
    print("Weather prediction:  ",predict_weather)
    new_row.append(predict_senti)
    new_row.append(predict_when)
    new_row.append(predict_weather)
    out.writerow(new_row)   
out1.close()
v1.close()
v.close()
    
    
    








#kinds of sentiments
s1 = "I can't tell"
s2 = "Negative"
s3 = "Neutral / author is just sharing information"
s4 = "Positive"
s5 = "Tweet not related to weather condition"
#kinds of when 
w1 = "current (same day) weather"
w2 = "future (forecast)"
w3 = "I can't tell"
w4 = "past weather"
#kinds of weather
k1 = "clouds"
k2 = "cold"
k3 = "dry"
k4 = "hot"
k5 = "humid"
k6 = "hurricane"
k7 = "I can't tell"
k8 = "ice"
k9 = "other"
k10 = "rain"
k11 = "snow"
k12 = "storms"
k13 = "sun"
k14 = "tornado"
k15 = "wind"

##import csv
##with open('sklearn_prediction.csv', 'r') as csvfile:
##    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
##    for row in spamreader:
##         print(row)
