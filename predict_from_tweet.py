import pickle
import pandas as p
import io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import MultiTaskLasso, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor 
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn import svm
from scipy import sparse
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
import time
import nltk
from nltk.tokenize import wordpunct_tokenize


classify = joblib.load('joblib_500.pkl')
tdf = classify[0]
lsa = classify[1]
norm = classify[2]
clf =  classify[3]

paths = ['test_small.csv']
t = p.read_csv(paths[0])

test_data = tdf.transform(t['tweet'].tolist())
print("test_data shape -->", test_data.shape)

test_data = lsa.transform(test_data)
print("test_data shape -->", test_data.shape)

test_data = norm.transform(test_data, copy= False)
print("test_data shape -->", test_data.shape)

prediction = np.array(clf.predict(test_data))

prediction = np.abs(prediction*(prediction > 0))
prediction[prediction > 1] = 1
prediction[prediction < 0] = 0
print("prediction", prediction)

prediction = np.array(np.hstack([np.matrix(t['id']).T, prediction])) 
col = '%i,' + '%f,'*23 + '%f'
np.savetxt('sklearn_prediction_new.csv', prediction,col, delimiter=',')

