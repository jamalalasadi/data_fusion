import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import svm, datasets
# from skimage import io, transform
import sklearn.metrics as met
from sklearn import svm
from sklearn.metrics import accuracy_score
import sklearn.metrics as mt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
import sys


def conf_mat_param(pred_labels,true_labels):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == -1, true_labels == -1))  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == -1))  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == -1, true_labels == 1)) 
    return (TP,FP,TN,FN)
    
    
    
filepath = '/home/jamal/Desktop/Research three/support vector /Final (1).csv'
col=['AUDIO_text_len','AUDIO_text_sentiment','AUDIO_percent_bad_words','AUDIO_valence','AUDIO_arousal','AUDIO_speech_percent','AUDIO_music_percent','ADIO_silence_percent','AUDIO_loudness','AUDIO_glove1','AUDIO_glove2','AUDIO_glove3','TEXT_text_len','TEXT_percent_punctuation','TEXT_percent_uppercase','TEXT_text_sentiment','TEXT_percent_bad_words','TEXT_valence','TEXT_arousal','TEXT_glove1','TEXT_glove2','TEXT_glove3','VISUAL_num_faces','VISUAL_valence','VISUAL_arousal','VISUAL_gore','VISUAL_explicit','VISUAL_drug','VISUAL_suggestive','VISUAL_ocr_len','VISUAL_labels1','VISUAL_labels2','VISUAL_labels3','Victim','bully']

Data = pd.read_csv(filepath,delimiter = ',',names = col)

from imblearn.over_sampling import SMOTE
y = Data['bully']
X = Data.drop('bully',axis = 1)
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
# 

audio_features = []
text_features = []
vedio_features = []
labels = []
Test_length = len(Data) - int(len(Data)*.15)
Test_Data = Data.loc[Test_length:]

iterations = np.random.permutation(len(Data))
train_iter = iterations[:Test_length]
test_iter = iterations[Test_length:]

# for i in train_iter:
#     lst = Data.loc[i].values.tolist()
#     audio_features.append(lst[0:12])
#     text_features.append(lst[12:22])
#     vedio_features.append(lst[22:-2])
#     if lst[-1]==0:
#         labels.append(-1)
#     if lst[-1]==1:
#         labels.append(1)

for i in train_iter:
    lst = [X_resampled[i],y_resampled[i]]
    audio_features.append(lst[0][0:12])
    text_features.append(lst[0][12:22])
    vedio_features.append(lst[0][22:-1])
    if lst[1]==0:
        labels.append(-1)
    if lst[1]==1:
        labels.append(1)
    

# clf_audio = svm.SVC(kernel = 'linear')# support vector machine model training 
# clf_audio.fit(audio_features,labels)
# 
# clf_text = svm.SVC(gamma = 2)# support vector machine model training 
# clf_text.fit(text_features,labels)
# 
# clt_vedio = svm.SVC(gamma = 2)# support vector machine model training 
# clt_vedio.fit(vedio_features,labels)

gnb_audio = GaussianNB()
clf_audio = gnb_audio.fit(audio_features, labels)
clf_audio_p = clf_audio.predict_proba(audio_features) #(new) calculate the probability of classes (audio)
clf_audio_b = clf_audio.predict(audio_features)

gnb_text = GaussianNB()

clf_text = gnb_text.fit(text_features, labels)
clf_text_p = clf_text.predict_proba(text_features) #(new) calculate the probabilities of classes (text)
clf_text_b = clf_text.predict(text_features)


gnb_vedio = GaussianNB()

clf_vedio = gnb_vedio.fit(vedio_features, labels)
clf_vedio_p = clf_vedio.predict_proba(vedio_features) #(new) calculate the probabilities of classes (vedio)
clf_vedio_b = clf_vedio.predict(vedio_features)


# Construct the average model (probability)
# A = np.array([clf_audio_p[:,0],clf_text_p[:,0],clf_vedio_p[:,0]])
# A = A.transpose()
# w = np.dot(np.linalg.pinv(A),labels)

# construct the average model (lablesl)
A = np.array([clf_audio_b,clf_text_b,clf_vedio_b])
A = A.transpose()
w = np.dot(np.linalg.pinv(A),labels)
# clf_audio =  LogisticRegression(  solver = 'lbfgs').fit(audio_features,labels)# logistic regression model for audio
# clf_text =  LogisticRegression(solver = 'lbfgs').fit(text_features,labels)# logistic regression model for text
# clf_vedio =  LogisticRegression( solver = 'lbfgs').fit(vedio_features,labels)# logistic regression model for vedio

# clf = neighbors.KNeighborsClassifier(10)# K  nearest neighbor model
# clf.fit(features,labels)
# gaussianNB_clf = GaussianNB()# Gaussian Naive Bayes model
# gaussianNB_clf.fit(features,labels)
# Random_clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)# Random Forest model
# Random_clf.fit(features,labels)
# # RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=2, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
#             oob_score=False, random_state=0, verbose=0, warm_start=False)
#             
#             
# text_features_test = []
# audio_features_test = []
# vedio_features_test = []
# labels_test = []
# 
# for i in range(Test_length,len(Data)):
#     lst = Test_Data.loc[i].tolist()
#     audio_features_test.append(lst[0:12])
#     text_features_test.append(lst[12:22])
#     vedio_features_test.append(lst[22:-2])
#     if lst[-2]==0:
#         labels_test.append(-1)
#     if lst[-2]==1:
#         labels_test.append(1)    
# 
# 
# 
# y_test_audio = clf_audio.predict(audio_features_test)# logistic regression model test for audio
# acc_audio = accuracy_score(labels_test,y_test_audio)
# conf_mat_audio = mt.confusion_matrix(labels_test,y_test_audio)
# print(conf_mat_audio)
# tn_audio, fp_audio, fn_audio, tp_audio = conf_mat_audio.ravel()
# TPR_audio = tp_audio/(tp_audio+fn_audio)
# FPR_audio = fp_audio/(fp_audio+tn_audio)


################################

text_features_test = []
audio_features_test = []
vedio_features_test = []



labels_test = []

# for i in test_iter:
#     lst = Data.loc[i].tolist()
#     
#     if lst[-1]==0:
#         
#         labels_test.append(-1)   #non-bully
#     else:
#         labels_test.append(1)    #bully
#     audio_features_test.append(lst[0:12])
#     text_features_test.append(lst[12:22])
#     vedio_features_test.append(lst[22:-2])
#     
    
for i in test_iter:
    lst = [X_resampled[i],y_resampled[i]]
    
    if lst[1]==0:
        
        labels_test.append(-1)   #non-bully
    else:
        labels_test.append(1)    #bully
    audio_features_test.append(lst[0][0:12])
    text_features_test.append(lst[0][12:22])
    vedio_features_test.append(lst[0][22:-1])
    
        
# y_test_audio = clf_audio.predict(audio_features_test)# logistic regression model test for audio
# acc_audio = accuracy_score(labels_test,y_test_audio)
# conf_mat_audio = mt.confusion_matrix(labels_test,y_test_audio)
# tn_audio, fp_audio, fn_audio, tp_audio = conf_mat_audio.ravel()
# TPR_audio = tp_audio/(tp_audio+fn_audio)
# FPR_audio = fp_audio/(fp_audio+tn_audio)

# y_test = SVM_clf.predict(features_test)# support vector machine model test
# acc = accuracy_score(labels_test,y_test)
# print('SVM')
# print(acc)
# print(mt.confusion_matrix(labels_test,y_test))
# print('precision score ',met.precision_score(labels_test,y_test))
# print('recall score', met.recall_score(labels_test,y_test))
# print('f1 score',met.f1_score(labels_test,y_test))


# labels_test_male = np.array(labels_test_male)

## Audio Male Female
y_test_audio = clf_audio.predict(audio_features_test)# logistic regression model test for audio
y_test_audio_p = np.array(clf_audio.predict_proba(audio_features_test))[:,0]

acc_audio = accuracy_score(labels_test,y_test_audio)
conf_audio = mt.confusion_matrix(labels_test,y_test_audio)
tn_audio, fp_audio, fn_audio, tp_audio = conf_audio.ravel()
print("audio ",acc_audio)
# TPR_audio_female = tp_audio/(tp_audio+fn_audio)
# FPR_audio_female = fp_audio/(fp_audio+tn_audio)

# Delta_TPR = TPR_audio_male - TPR_audio_female

# print('logistic regression for audio')
# print(acc_audio)
# print(mt.confusion_matrix(labels_test,y_test_audio))
# print('precision score audio',met.precision_score(labels_test,y_test_audio))
# print('recall score audio', met.recall_score(labels_test,y_test_audio))
# print('f1 score audio',met.f1_score(labels_test,y_test_audio))


## Text Male Female
y_test_text = clf_text.predict(text_features_test)# logistic regression model test for audio
y_test_text_p = np.array(clf_text.predict_proba(text_features_test))[:,0]
acc_text = accuracy_score(labels_test,y_test_text)
# conf_text = mt.confusion_matrix(labels_test,y_test_text)
# tn_text, fp_text, fn_text, tp_text = conf_text.ravel()
print("text ",acc_text)
# print('logistic regression for audio')
# print(acc_audio)
# print(mt.confusion_matrix(labels_test,y_test_audio))
# print('precision score audio',met.precision_score(labels_test,y_test_audio))
# print('recall score audio', met.recall_score(labels_test,y_test_audio))
# print('f1 score audio',met.f1_score(labels_test,y_test_audio))


# TPR_audio_male = tp_audio/(tp_audio+fn_audio)
# FPR_audio_male = fp_audio/(fp_audio+tn_audio)
# Delta_male = TPR_audio_male-FPR_audio_male

## Vedio Male Female
y_test_vedio = clf_vedio.predict(vedio_features_test)# logistic regression model test for audio
y_test_vedio_p = np.array(clf_vedio.predict_proba(vedio_features_test))[:,0]# logistic regression model test for audio

acc_vedio = accuracy_score(labels_test,y_test_vedio)
# conf_vedio = mt.confusion_matrix(labels_test,y_test_vedio)
# tn_vedio, fp_vedio, fn_vedio, tp_vedio = conf_vedio.ravel()
print("vedio ",acc_vedio)


############## Majority voting
y_majority = y_test_text+y_test_audio+y_test_vedio
y_majority [y_majority>0] = 1
y_majority [y_majority<0] = -1
acc_majority = accuracy_score(labels_test,y_majority)
print("majority",acc_majority)

####### Average Voting (probabilities)
p_average = (y_test_audio_p+y_test_text_p+y_test_vedio_p)/3.0 # finding the probabilities
y_p = p_average.copy()
y_p[y_p<.5] = 1
y_p[y_p>=.5] = -1
acc_p = accuracy_score(labels_test,y_p)
print("Average Probability acc:",acc_p)

### least square weights
A_p = np.array([[y_test_audio_p],[y_test_text_p],[y_test_vedio_p]])
A_p = A_p.transpose()
y_p2 = np.dot(A_p,w) 
y_pfinal = y_p2.copy()
y_pfinal[y_p2>0] = -1
y_pfinal[y_p2<=0] = 1
acc_p2 = accuracy_score(labels_test,y_pfinal)
print("weighted Probability acc:",acc_p2)
# TPR_audio_female = tp_audio/(tp_audio+fn_audio)
# FPR_audio_female = fp_audio/(fp_audio+tn_audio)

# Delta_TPR = TPR_audio_male - TPR_audio_female

# print('logistic regression for audio')
# print(acc_audio)
# print(mt.confusion_matrix(labels_test,y_test_audio))
# print('precision score audio',met.precision_score(labels_test,y_test_audio))
# print('recall score audio', met.recall_score(labels_test,y_test_audio))
# print('f1 score audio',met.f1_score(labels_test,y_test_audio))


# y_test_text = clf_text.predict(text_features_test)# logistic regression model test for text
# acc_text = accuracy_score(labels_test,y_test_text)
# print('logistic regression for text')
# print(acc_text)
# print(mt.confusion_matrix(labels_test,y_test_text))
# print('precision score text',met.precision_score(labels_test,y_test_text))
# print('recall score text', met.recall_score(labels_test,y_test_text))
# print('f1 score text',met.f1_score(labels_test,y_test_text))
# 
# 
# 
# 
# y_test_vedio = clf_vedio.predict(vedio_features_test)# logistic regression model test
# acc_vedio = accuracy_score(labels_test,y_test_vedio)
# print('logistic regression for vedio')
# print(acc_vedio)
# print(mt.confusion_matrix(labels_test,y_test_vedio))
# print('precision score vedio',met.precision_score(labels_test,y_test_vedio))
# print('recall score vedio', met.recall_score(labels_test,y_test_vedio))
# print('f1 score vedio',met.f1_score(labels_test,y_test_vedio))
# 


# y_test = clf.predict(features_test) # K _ nearest neighbor model test
# acc = accuracy_score(labels_test,y_test)
# print('k nearest neighbor')
# print(acc)
# print(mt.confusion_matrix(labels_test,y_test))
# print('precision score ',met.precision_score(labels_test,y_test))
# print('recall score', met.recall_score(labels_test,y_test))
# print('f1 score',met.f1_score(labels_test,y_test))
# 
# 
# y_test = gaussianNB_clf.predict(features_test)# Gaussian Naive Bayes model test
# acc = accuracy_score(labels_test,y_test)
# print('Gaussian bayesian model')
# print(acc)
# print(mt.confusion_matrix(labels_test,y_test))
# print('precision score ',met.precision_score(labels_test,y_test))
# print('recall score', met.recall_score(labels_test,y_test))
# print('f1 score',met.f1_score(labels_test,y_test))
# 
# 
# 
# y_test = Random_clf.predict(features_test)# Random Forest model test
# acc = accuracy_score(labels_test,y_test)
# print('Random Forest')
# print(acc)
# print(mt.confusion_matrix(labels_test,y_test))
# print('precision score ',met.precision_score(labels_test,y_test))
# print('recall score', met.recall_score(labels_test,y_test))
# print('f1 score',met.f1_score(labels_test,y_test))

#########
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.svm import SVC
# 
# # Create adaboost classifer object
# svc= LogisticRegression(solver = 'lbfgs' )
# 
# 
# abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=2)
# 
# all_features = audio_features
# all_labels = labels
# # Train Adaboost Classifer
# model = abc.fit(np.array(all_features), np.array(all_labels))
# 
# #Predict the response for test dataset
# y_pred_female = model.predict(audio_features_test_female)
# y_pred_male = model.predict(audio_features_test_male)
# 
# print("Audio female all:",metrics.accuracy_score(y_pred_female, labels_test_female))
# print("Audio male all:",metrics.accuracy_score(y_pred_male, labels_test_male))
# 
# 
# ##############3
# svc= LogisticRegression( solver = 'lbfgs' )
# 
# abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=2)
# 
# all_features = vedio_features
# all_labels = labels
# # Train Adaboost Classifer
# model1 = abc.fit(np.array(all_features), np.array(all_labels))
# 
# #Predict the response for test dataset
# y_pred_female = model1.predict(vedio_features_test_female)
# y_pred_male = model1.predict(vedio_features_test_male)
# 
# print("vedio female all:",metrics.accuracy_score(y_pred_female, labels_test_female))
# print("vedio male all:",metrics.accuracy_score(y_pred_male, labels_test_male))
# 
# #############
# svc=LogisticRegression(solver = 'lbfgs' ,max_iter=10000 )
# 
# abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=2)
# 
# all_features = text_features
# all_labels = labels
# # Train Adaboost Classifer
# model1 = abc.fit(np.array(all_features), np.array(all_labels))
# 
# #Predict the response for test dataset
# y_pred_female = model1.predict(text_features_test_female)
# y_pred_male = model1.predict(text_features_test_male)
# 
# print("text female all:",metrics.accuracy_score(y_pred_female, labels_test_female))
# print("tex male all:",metrics.accuracy_score(y_pred_male, labels_test_male))
# ################################################# SMOTE
# print('-----------------------------------------------------------------------------')
# from imblearn.over_sampling import SMOTE
# y = Data['bully']
# X = Data.drop('bully',axis = 1)
# X_resampled, y_resampled = SMOTE().fit_resample(X, y)
# 
# 
# Data = pd.read_csv(filepath,delimiter = ',',names = col)
# audio_features = []
# text_features = []
# vedio_features = []
# labels = []
# Test_length = len(X_resampled) - int(len(X_resampled)*.4)
# Test_Data = X_resampled[Test_length:]
# 
# labels_test_male = []
# labels_test_female = []
# 
# text_features_test_male = []
# audio_features_test_male = []
# vedio_features_test_male = []
# 
# text_features_test_female = []
# audio_features_test_female = []
# vedio_features_test_female = []
# iterations = np.random.permutation(len(X_resampled))
# train_iter = iterations[:Test_length]
# test_iter = iterations[Test_length:]
# # Train
# for i in train_iter:
#     lst = [X_resampled[i],y_resampled[i]]
#     audio_features.append(lst[0][0:12])
#     text_features.append(lst[0][12:22])
#     vedio_features.append(lst[0][22:-1])
#     if lst[1]==0:
#         labels.append(-1)
#     if lst[1]==1:
#         labels.append(1)
#         
# clf_audio =  LogisticRegression(solver = 'lbfgs').fit(audio_features,labels)# logistic regression model for audio
# clf_text =  LogisticRegression( solver = 'lbfgs').fit(text_features,labels)# logistic regression model for text
# clf_vedio =  LogisticRegression(solver = 'lbfgs').fit(vedio_features,labels)# logistic regression model for vedio
#        
# ######################################## Test
#         
# for i in test_iter:
#     lst = [X_resampled[i],y_resampled[i]]
#     
#     if lst[0][-1]==0:
#         if lst[1]==0:
#             labels_test_female.append(-1)   #non-bully
#         else:
#             labels_test_female.append(1)    #bully
#         audio_features_test_female.append(lst[0][0:12])
#         text_features_test_female.append(lst[0][12:22])
#         vedio_features_test_female.append(lst[0][22:-1])
#     if lst[0][-1]==0:
#         if lst[1]==0:
#             labels_test_male.append(-1)   #non-bully
#         else:
#             labels_test_male.append(1)
#         audio_features_test_male.append(lst[0][0:12])
#         text_features_test_male.append(lst[0][12:22])
#         vedio_features_test_male.append(lst[0][22:-1])
# 
# 
# 
# 
# ## Audio Male Female
# y_test_audio_female = clf_audio.predict(audio_features_test_female)# logistic regression model test for audio
# acc_audio = accuracy_score(labels_test_female,y_test_audio_female)
# conf_female_audio = mt.confusion_matrix(labels_test_female,y_test_audio_female)
# tn_audio, fp_audio, fn_audio, tp_audio = conf_female_audio.ravel()
# print("audio female",acc_audio)
# # TPR_audio_female = tp_audio/(tp_audio+fn_audio)
# # FPR_audio_female = fp_audio/(fp_audio+tn_audio)
# 
# # Delta_TPR = TPR_audio_male - TPR_audio_female
# 
# # print('logistic regression for audio')
# # print(acc_audio)
# # print(mt.confusion_matrix(labels_test,y_test_audio))
# # print('precision score audio',met.precision_score(labels_test,y_test_audio))
# # print('recall score audio', met.recall_score(labels_test,y_test_audio))
# # print('f1 score audio',met.f1_score(labels_test,y_test_audio))
# 
# y_test_audio_male = clf_audio.predict(audio_features_test_male)# logistic regression model test for audio
# acc_audio = accuracy_score(labels_test_male,y_test_audio_male)
# conf_male_audio = mt.confusion_matrix(labels_test_male,y_test_audio_male)
# tn_audio, fp_audio, fn_audio, tp_audio = conf_male_audio.ravel()
# TPR_audio_male = tp_audio/(tp_audio+fn_audio)
# FPR_audio_male = fp_audio/(fp_audio+tn_audio)
# Delta_male = TPR_audio_male-FPR_audio_male
# print(Delta_male)
# print("audio male",acc_audio)
# 
# ## Text Male Female
# y_test_text_female = clf_text.predict(text_features_test_female)# logistic regression model test for audio
# acc_text = accuracy_score(labels_test_female,y_test_text_female)
# conf_female_text = mt.confusion_matrix(labels_test_female,y_test_text_female)
# tn_audio, fp_audio, fn_audio, tp_audio = conf_female_text.ravel()
# TPR_audio_female = tp_audio/(tp_audio+fn_audio)
# FPR_audio_female = fp_audio/(fp_audio+tn_audio)
# TPR_audio_male = tp_audio/(tp_audio+fn_audio)
# FPR_audio_male = fp_audio/(fp_audio+tn_audio)
# Delta_male = TPR_audio_male-FPR_audio_male
# print(Delta_male)
# print("text female",acc_text)
# # print('logistic regression for audio')
# # print(acc_audio)
# # print(mt.confusion_matrix(labels_test,y_test_audio))
# # print('precision score audio',met.precision_score(labels_test,y_test_audio))
# # print('recall score audio', met.recall_score(labels_test,y_test_audio))
# # print('f1 score audio',met.f1_score(labels_test,y_test_audio))
# 
# y_test_text_male = clf_text.predict(text_features_test_male)# logistic regression model test for audio
# acc_text = accuracy_score(labels_test_male,y_test_text_male)
# conf_male_text = mt.confusion_matrix(labels_test_male,y_test_text_male)
# tn_audio, fp_audio, fn_audio, tp_audio = conf_male_text.ravel()
# TPR_audio_male = tp_audio/(tp_audio+fn_audio)
# FPR_audio_male = fp_audio/(fp_audio+tn_audio)
# Delta_male = TPR_audio_male-FPR_audio_male
# print(Delta_male)
# print("text male",acc_text)
# # TPR_audio_male = tp_audio/(tp_audio+fn_audio)
# # FPR_audio_male = fp_audio/(fp_audio+tn_audio)
# # Delta_male = TPR_audio_male-FPR_audio_male
# 
# ## Vedio Male Female
# y_test_vedio_female = clf_vedio.predict(vedio_features_test_female)# logistic regression model test for audio
# acc_vedio = accuracy_score(labels_test_female,y_test_vedio_female)
# conf_female_vedio = mt.confusion_matrix(labels_test_female,y_test_vedio_female)
# tn_audio, fp_audio, fn_audio, tp_audio = conf_female_vedio.ravel()
# print("vedio female",acc_vedio)
# # TPR_audio_female = tp_audio/(tp_audio+fn_audio)
# # FPR_audio_female = fp_audio/(fp_audio+tn_audio)
# 
# # Delta_TPR = TPR_audio_male - TPR_audio_female
# 
# # print('logistic regression for audio')
# # print(acc_audio)
# # print(mt.confusion_matrix(labels_test,y_test_audio))
# # print('precision score audio',met.precision_score(labels_test,y_test_audio))
# # print('recall score audio', met.recall_score(labels_test,y_test_audio))
# # print('f1 score audio',met.f1_score(labels_test,y_test_audio))
# 
# y_test_vedio_male = clf_vedio.predict(vedio_features_test_male)# logistic regression model test for audio
# acc_vedio = accuracy_score(labels_test_male,y_test_vedio_male)
# conf_male_vedio = mt.confusion_matrix(labels_test_male,y_test_vedio_male)
# tn_audio, fp_audio, fn_audio, tp_audio = conf_male_vedio.ravel()
# print("vedio male",acc_vedio)
# 
# 
# 
# 
# svc= LogisticRegression(solver = 'lbfgs' )
# 
# abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=2)
# 
# all_features = audio_features
# all_labels = labels
# # Train Adaboost Classifer
# model = abc.fit(np.array(all_features), np.array(all_labels))
# 
# #Predict the response for test dataset
# y_pred_female = model.predict(audio_features_test_female)
# y_pred_male = model.predict(audio_features_test_male)
# 
# print("Audio female all:",metrics.accuracy_score(y_pred_female, labels_test_female))
# print("Audio male all:",metrics.accuracy_score(y_pred_male, labels_test_male))
# 
# 
# ##############3
# svc= LogisticRegression( solver = 'lbfgs' )
# 
# abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=2)
# 
# all_features = vedio_features
# all_labels = labels
# # Train Adaboost Classifer
# model1 = abc.fit(np.array(all_features), np.array(all_labels))
# 
# #Predict the response for test dataset
# y_pred_female = model1.predict(vedio_features_test_female)
# y_pred_male = model1.predict(vedio_features_test_male)
# 
# print("vedio female all:",metrics.accuracy_score(y_pred_female, labels_test_female))
# print("vedio male all:",metrics.accuracy_score(y_pred_male, labels_test_male))
# 
# #############
# svc=LogisticRegression(solver = 'lbfgs',max_iter=10000 )
# 
# abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=2)
# 
# all_features = text_features
# all_labels = labels
# # Train Adaboost Classifer
# model1 = abc.fit(np.array(all_features), np.array(all_labels))
# 
# #Predict the response for test dataset
# y_pred_female = model1.predict(text_features_test_female)
# y_pred_male = model1.predict(text_features_test_male)
# 
# print("text female all:",metrics.accuracy_score(y_pred_female, labels_test_female))
# print("tex male all:",metrics.accuracy_score(y_pred_male, labels_test_male))
# 