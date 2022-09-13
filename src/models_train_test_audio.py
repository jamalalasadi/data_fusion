import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import svm, datasets
from skimage import io, transform
import sklearn.metrics as met
from sklearn import svm
from sklearn.metrics import accuracy_score
import sklearn.metrics as mt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score


filepath = '/home/jamal/Documents/support vector /Re%3a_Cyberbullying_dataset/audio_train_df_new.csv'
col = ['AUDIO_text_len','AUDIO_text_sentiment','AUDIO_percent_bad_words','AUDIO_valence','AUDIO_arousal',	'AUDIO_speech_percent','AUDIO_music_percent','AUDIO_silence_percent','AUDIO_loudness','AUDIO_glove1','AUDIO_glove2 ','AUDIO_glove3' , 'id',	'bully']

Data = pd.read_csv(filepath,delimiter = ',',names = col)
features = []
labels = []
for i in range(len(Data)):
    lst = Data.loc[i].tolist()
    features.append(lst[0:-2])
    if lst[-1]=='bullying':
        labels.append(1)
    if lst[-1]=='noneBll':
        labels.append(0)
    

SVM_clf = svm.SVC(gamma=2)# support vector machine model training 
SVM_clf.fit(features,labels)
clf2 =  LogisticRegression().fit(features,labels)# logistic regression model
clf = neighbors.KNeighborsClassifier(10)# K  nearest neighbor model
clf.fit(features,labels)
gaussianNB_clf = GaussianNB()# Gaussian Naive Bayes model
gaussianNB_clf.fit(features,labels)
Random_clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)# Random Forest model
Random_clf.fit(features,labels)
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=2, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
#             oob_score=False, random_state=0, verbose=0, warm_start=False)
#             
#             
            
################################

filepath_test = '/home/jamal/Documents/support vector /Re%3a_Cyberbullying_dataset/audio_test_df_new.csv'
Data_test = pd.read_csv(filepath_test,delimiter = ',',names = col)
features_test = []
labels_test = []
for i in range(len(Data_test)):
    lst = Data_test.loc[i].tolist()
    features_test.append(lst[0:-2])
    if lst[-1]=='bullying':
        labels_test.append(1)
    if lst[-1]=='noneBll':
        labels_test.append(0)
        

y_test = SVM_clf.predict(features_test)# support vector machine model test
acc = accuracy_score(labels_test,y_test)
print('SVM')
print(acc)
print(mt.confusion_matrix(labels_test,y_test))
print('precision score ',met.precision_score(labels_test,y_test))
print('recall score', met.recall_score(labels_test,y_test))
print('f1 score',met.f1_score(labels_test,y_test))



y_test = clf2.predict(features_test)# logistic regression model test
acc = accuracy_score(labels_test,y_test)
print('logistic regression')
print(acc)
print(mt.confusion_matrix(labels_test,y_test))
print('precision score ',met.precision_score(labels_test,y_test))
print('recall score', met.recall_score(labels_test,y_test))
print('f1 score',met.f1_score(labels_test,y_test))


y_test = clf.predict(features_test) # K _ nearest neighbor model test
acc = accuracy_score(labels_test,y_test)
print('k nearest neighbor')
print(acc)
print(mt.confusion_matrix(labels_test,y_test))
print('precision score ',met.precision_score(labels_test,y_test))
print('recall score', met.recall_score(labels_test,y_test))
print('f1 score',met.f1_score(labels_test,y_test))


y_test = gaussianNB_clf.predict(features_test)# Gaussian Naive Bayes model test
acc = accuracy_score(labels_test,y_test)
print('Gaussian bayesian model')
print(acc)
print(mt.confusion_matrix(labels_test,y_test))
print('precision score ',met.precision_score(labels_test,y_test))
print('recall score', met.recall_score(labels_test,y_test))
print('f1 score',met.f1_score(labels_test,y_test))



y_test = Random_clf.predict(features_test)# Random Forest model test
acc = accuracy_score(labels_test,y_test)
print('Random Forest')
print(acc)
print(mt.confusion_matrix(labels_test,y_test))
print('precision score ',met.precision_score(labels_test,y_test))
print('recall score', met.recall_score(labels_test,y_test))
print('f1 score',met.f1_score(labels_test,y_test))

#########
