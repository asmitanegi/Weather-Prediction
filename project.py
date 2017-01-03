import pickle
import time
import sys
import select
import h5py
import pandas as p
import numpy as np
import io
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import MultiTaskLasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import LinearSVC
from sklearn import svm
from scipy import sparse
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
from sklearn.linear_model import SGDRegressor
import nltk
from nltk.tokenize import wordpunct_tokenize

###################################################################################################################################################

##class Util:
##
##    def create_t_matrix(self, y):
##        classes = np.max(y)
##        t = np.zeros((y.shape[0], classes+1))
##        for i in range(y.shape[0]):
##            t[np.int32(i), np.int32(y[i])] = 1.0
##            
##        return t
##
##u = Util()
   
###################################################################################################################################################

def pathInitialization():
    global trainData,testData
    trainData = p.read_csv('train.csv')
    testData = p.read_csv('test.csv')

###################################################################################################################################################

    # Both the tokenizers are efficient for tweet analysis as they do not remove
    # the punctuations and special characters. Going with the popular trend of
    # using emocations to showcase once emotions can be the key for sentiment 
    # analysis.

class SnowballTokenizer(object):
    def __init__(self):
        self.wnl = nltk.stem.SnowballStemmer("english")
    def __call__(self, doc):
        return [self.wnl.stem(t) for t in wordpunct_tokenize(doc)] 
    
class WordnetTokenizer(object):
    def __init__(self):
        self.wnl = nltk.stem.WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in wordpunct_tokenize(doc)]   

###################################################################################################################################################

    #Method to select classifier with different configuration
    #option is there to select a classifier but we have found
    #the ridge to be the best and hence used by default 


def selectClassifier(choice):
    if choice == 1:
        classifier = ExtraTreesRegressor(max_depth = 25, n_estimators = 28, max_features = 100, n_jobs = 5)
    if choice == 2:
        classifier = Ridge()
    if choice == 3:
        classifier = RandomForestRegressor(max_depth = 30, n_estimators = 30, max_features = 100, n_jobs = 5)

    return classifier

###################################################################################################################################################

    # Different TFIDF Vectorizers selected after carefull selection procedure
    # We have used 5 different configuration which we deemed useful to the get
    # a better result for tweet vectorization task
        
def selectVectorizer(choice):

    if choice == 1:     
        vector = TfidfVectorizer(max_features=80000, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',sublinear_tf=1,
                                   ngram_range=(1, 3), tokenizer = WordnetTokenizer())            
    if choice == 2:
        vector = TfidfVectorizer(max_features=10500, strip_accents='unicode', stop_words = 'english', analyzer='char',sublinear_tf=1, ngram_range=(2, 17))

    if choice == 3:
        vector = CountVectorizer(max_features=51000, strip_accents='unicode', analyzer='word',token_pattern=r'\w{2,}', ngram_range=(1, 3),
                                   tokenizer = SnowballTokenizer())

    if choice == 4:
        vector = CountVectorizer(max_features=25000, strip_accents='unicode', stop_words = 'english',analyzer='char', ngram_range=(2, 6))

    if choice == 5:
        vector = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='char_wb',sublinear_tf=1,ngram_range=(2, 9))

    return vector
    
###################################################################################################################################################


    # This is the main function of the code, all the initilization,
    # classifier and vectorization class selction, dimentionality
    # reduction is done here

def main():
    global trainData, testData
    heldOut_pred_list = []
    test_pred_list = []

    #Make this 1, if want to perfor the dimentionality reduction
    
    use_lsa = 1
    pathInitialization()
    print("traindata is :", len(trainData))
    print("testData is :", len(testData))

    # Choose data% as held out set this is used to split the
    # percentage of data as training and heldout set
    
    division = 0.20

    n = len(trainData)
    train_last = int(np.round(n * (1-division)))
    heldOut_Start = int(np.round((1-division)* n))

    y = np.array(trainData.ix[:,4:])
    trainTweets = trainData['tweet'].tolist()[0:train_last]    

    heldOut_Tweets = np.array(trainData['tweet'].tolist()[heldOut_Start:])
    cv_y = np.array(y[heldOut_Start:])
    #print("heldOut_Tweets len --> ",len(heldOut_Tweets))
    #print("cv_y len -->",len(cv_y))

    if division == 0:
        trainTweets = trainData['tweet'].tolist()
    else:
        y = y[0:int(np.round(len(trainData['tweet'].tolist())*(1-division)))]
        #print("y len is ",len(y))

    
    # Here different vectors can be selected from the above mentioned
    # configurations

    tfid = selectVectorizer(5)
    clf = selectClassifier(2)
    
    print('FITTING VECTORIZER...')
    testTweets = testData['tweet'].tolist()
    tfid.fit(trainTweets + testTweets)
    
    #tfid.fit(trainData['tweet'].tolist() + testData['tweet'].tolist())
    
    print('TRANSFORMING TRAIN TWEETS ...')
    X = tfid.transform(trainTweets)
    print("X shape ", X.shape)

    print('TRANSFORMING...')    
    heldOut_X = tfid.transform(heldOut_Tweets)
    print("heldOut_X shape is ",heldOut_X.shape)

    print('TRANSFORMING TEST TWEETS ...')    
    testTweets = tfid.transform(testTweets)
    print("TEST TWEET SHAPE IS ",testTweets.shape)      

    
    # Choose any classifier from the above mentioned configuration
   
    if use_lsa == 1:
        print("DIMENTIONALITY REDUCTION --> ")  
        LSA = TruncatedSVD(n_components = 500)
        print('FITTING LSA --> ')
        LSA.fit(X,y)
        print ('TRANSFORMING LSA --> ')
        X = LSA.transform(X)
        print("X shape is --> ",X.shape)
        
        heldOut_X = LSA.transform(heldOut_X)
        print("heldOut_X shape is --> ",heldOut_X.shape)
        testTweets = LSA.transform(testTweets)
        print("testTweets shape is --> ",testTweets.shape)
        
        print('FITTING NORMALIZING --> ')
        normalize = Normalizer()
        normalize.fit(X,y)

        print("TRANSFORMING NORMALIZING")
        X = normalize.transform(X, copy= False)
        print("X shape is --> ",X.shape)
        heldOut_X = normalize.transform(heldOut_X, copy= False)
        testTweets = normalize.transform(testTweets, copy= False)
        print("TEST TWEET SHAPE IS --> ",testTweets.shape)
        
    else:
        print("HOT VECTOR")
        fac = p.Categorical(trainData['state'].tolist()+ testData['state'].tolist())
        t_matrix = u.create_t_matrix(fac.labels)
        train_feat = t_matrix[0:train_last]
        heldOut_X_feat = t_matrix[heldOut_Start:n]
        test_feat = t_matrix[n:]                        
        X = sparse.hstack([X, sparse.csr_matrix(train_feat)])
        heldOut_X = sparse.hstack([heldOut_X, sparse.csr_matrix(heldOut_X_feat)])
        testTweets = sparse.hstack([testTweets, sparse.csr_matrix(test_feat)])
        print("TEST TWEET SHAPE IS --> ",testTweets.shape)
  
    t0 = time.time()
        
    print('FITTING CLASSIFIERS --> ')
    print(X.shape)
    print(y.shape)
    clf.fit(X,y)
    
    print('VALIDATING --> ')

    train_error = clf.predict(X)-y
    train_error = (np.array(train_error)**2)
    train_error = np.array(train_error)
    train_error = (np.sqrt(np.sum(train_error)/ (X.shape[0]*24.0)))
    print('TRAIN ERROR: {0}'.format(train_error))
    print("PREDICTING TEST ")
    
    test_pred = np.array(clf.predict(testTweets))
    print("STEP1")
    test_pred = np.abs(test_pred*(test_pred > 0))
    test_pred[test_pred > 1] = 1
    test_pred[test_pred < 0] = 0

    
    heldOut_pred = np.array(clf.predict(heldOut_X))         
    heldOut_pred = np.abs(heldOut_pred*(heldOut_pred > 0))
    heldOut_pred[heldOut_pred > 1] = 1
    heldOut_pred[heldOut_pred < 0] = 0
    
    #CV test_pred
    heldOut_pred_list.append(heldOut_pred)
    
    #Test test_pred
    test_pred_list.append(test_pred)
    heldOut_error = heldOut_pred-cv_y
    heldOut_error = (np.array(heldOut_error)**2)
    heldOut_error = (np.sqrt(np.sum(heldOut_error)/ (heldOut_X.shape[0]*24.0)))
    print('CROSS VALIDATION ERROR : {0}'.format(heldOut_error))
            
    test_pred = np.array(np.hstack([np.matrix(testData['id']).T, test_pred])) 
    col = '%i,' + '%f,'*23 + '%f'
    np.savetxt('sklearn_prediction.csv', test_pred,col, delimiter=',')

    joblib.dump([tfid,LSA,normalize,clf], 'joblib_500.pkl')
   
    
##    heldOut_pred_list.append(cv_y)
##    pickle.dump(heldOut_pred_list, io.open('predicts.txt','wb'))
##    pickle.dump(test_pred_list, io.open('predicts_test.txt','wb'))

if __name__=="__main__":
    main()
