import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import svm, datasets
# from skimage import io, transform
import sklearn.metrics as met
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

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
import sys


def conf_mat_param(pred_labels,true_labels):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == -1, true_labels == -1))  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == -1))  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == -1, true_labels == 1))
    return (TP,FP,TN,FN)


ratio = 0.6  # the training testing ratio

filepath_m = '/home/jamal/Desktop/Research three/support vector /male.csv' # male dataset
filepath_f = '/home/jamal/Desktop/Research three/support vector /female 84.csv' # female dataset

col=['AUDIO_text_len','AUDIO_text_sentiment','AUDIO_percent_bad_words','AUDIO_valence','AUDIO_arousal','AUDIO_speech_percent','AUDIO_music_percent','ADIO_silence_percent','AUDIO_loudness','AUDIO_glove1','AUDIO_glove2','AUDIO_glove3','TEXT_text_len','TEXT_percent_punctuation','TEXT_percent_uppercase','TEXT_text_sentiment','TEXT_percent_bad_words','TEXT_valence','TEXT_arousal','TEXT_glove1','TEXT_glove2','TEXT_glove3','VISUAL_num_faces','VISUAL_valence','VISUAL_arousal','VISUAL_gore','VISUAL_explicit','VISUAL_drug','VISUAL_suggestive','VISUAL_ocr_len','VISUAL_labels1','VISUAL_labels2','VISUAL_labels3','Victim','bully']

Data_male = pd.read_csv(filepath_m,delimiter = ',',names = col)
Data_female = pd.read_csv(filepath_f,delimiter = ',',names = col)


# female accuracies variables

acc_audio_female = []
acc_text_female = []
acc_vedio_female = []
acc_test_audio_female = []
acc_test_text_female = []
acc_test_vedio_female = []
acc_majority_female = []
y_p_average_female_acc = []
Test_length_female = len(Data_female) - int(len(Data_female)*ratio)

auc_test_audio_female = []
auc_test_text_female = []
auc_test_vedio_female = []
auc_majority_female = []
y_p_average_female_auc = []

# male accuracies variables

acc_audio_male = []
acc_text_male = []
acc_vedio_male = []
acc_test_audio_male = []
acc_test_text_male = []
acc_test_vedio_male = []
acc_majority_male = []
y_p_average_male_acc = []
Test_length_male = len(Data_male) - int(len(Data_male)*ratio)

auc_test_audio_male = []
auc_test_text_male = []
auc_test_vedio_male = []
auc_majority_male = []
y_p_average_male_auc = []

# all accuracy variables
acc_audio_all = []
acc_text_all = []
acc_vedio_all = []
acc_test_audio_all = []
acc_test_text_all = []
acc_test_vedio_all = []

auc_test_audio_all = []
auc_test_text_all = []
auc_test_vedio_all = []


number_iter = 200 #change this to change the number of iterations you want to execute

for iter in range(number_iter):

    audio_features_all = [] # store audio features for male and female
    vedio_features_all = []
    text_features_all = []
    train_labels_all = []

    # Finding male training and testing indices
    iterations_male = np.random.permutation(len(Data_male))
    train_iter_male = iterations_male[:Test_length_male]
    test_iter_male = iterations_male[Test_length_male:]

    # Finding female training and testing indices
    iterations_female = np.random.permutation(len(Data_female))
    train_iter_female = iterations_female[:Test_length_female]
    test_iter_female = iterations_female[Test_length_female:]


    # Training phase
    audio_features_male = []
    text_features_male = []
    vedio_features_male = []
    lable_train_male = []

    # Extracting features from male dataset
    for i in train_iter_male:
        lst = Data_male.iloc[i].values.tolist()
        audio_features_all.append(lst[0:12])
        audio_features_male.append(lst[0:12])

        text_features_all.append(lst[12:22])
        text_features_male.append(lst[12:22])

        vedio_features_all.append(lst[22:-2])
        vedio_features_male.append(lst[22:-2])

        if lst[-1]==0:
            train_labels_all.append(-1)
            lable_train_male.append(-1)
        if lst[-1]==1:
            train_labels_all.append(1)
            lable_train_male.append(1)



    audio_features_female = []
    text_features_female = []
    vedio_features_female = []
    lable_train_female = []


    # Extracting features from female dataset
    for i in train_iter_female:
        lst = Data_female.iloc[i].values.tolist()
        audio_features_all.append(lst[0:12])
        audio_features_female.append(lst[0:12])

        text_features_all.append(lst[12:22])
        text_features_female.append(lst[12:22])

        vedio_features_all.append(lst[22:-2])
        vedio_features_female.append(lst[22:-2])

        if lst[-1]==0:
            train_labels_all.append(-1)
            lable_train_female.append(-1)
        if lst[-1]==1:
            train_labels_all.append(1)
            lable_train_female.append(1)


    # Training and find accuracies for audio features
    clf_audio =  GaussianNB().fit(audio_features_all,train_labels_all)# Classifier training for audio

    acc_audio_all.append(accuracy_score(train_labels_all,clf_audio.predict(audio_features_all))) # accuracy for all

    acc_audio_male.append(accuracy_score(lable_train_male,clf_audio.predict(audio_features_male))) # male accuracey

    acc_audio_female.append(accuracy_score(lable_train_female,clf_audio.predict(audio_features_female))) # female accuracey


    w_audio = acc_audio_all[-1]-(acc_audio_male[-1]-acc_audio_female[-1])

    # Training and finding accuracies for text features
    clf_text =  GaussianNB().fit(text_features_all,train_labels_all) # Calssifier training for text

    acc_text_all.append(accuracy_score(train_labels_all,clf_text.predict(text_features_all))) # accuracy for all

    acc_text_male.append(accuracy_score(lable_train_male,clf_text.predict(text_features_male))) # male accuracey

    acc_text_female.append(accuracy_score(lable_train_female,clf_text.predict(text_features_female))) # female accuracey

    w_text = acc_text_all[-1]-(acc_text_male[-1]-acc_text_female[-1])


    # Training and finding accuracies for vedio features


    clf_vedio =  GaussianNB().fit(vedio_features_all,train_labels_all)# Classifier training for vedio

    acc_vedio_all.append(accuracy_score(train_labels_all,clf_vedio.predict(vedio_features_all))) # accuracy for all

    acc_vedio_male.append( accuracy_score(lable_train_male,clf_vedio.predict(vedio_features_male))) # male accuracy

    acc_vedio_female.append(accuracy_score(lable_train_female,clf_vedio.predict(vedio_features_female))) # female accuracey


    w_vedio = acc_vedio_all[-1]-(acc_vedio_male[-1]-acc_vedio_female[-1])



    ################################ Testing

    text_features_test_male = []
    audio_features_test_male = []
    vedio_features_test_male = []

    text_features_test_female = []
    audio_features_test_female = []
    vedio_features_test_female = []

    text_features_test_all = []
    audio_features_test_all = []
    vedio_features_test_all = []



    labels_test_male = []
    labels_test_female = []
    labels_test_all = []


    # Extracting features from male dataset
    for i in test_iter_male:
        lst = Data_male.iloc[i].values.tolist()
        audio_features_test_all.append(lst[0:12])
        audio_features_test_male.append(lst[0:12])

        text_features_test_all.append(lst[12:22])
        text_features_test_male.append(lst[12:22])

        vedio_features_test_all.append(lst[22:-2])
        vedio_features_test_male.append(lst[22:-2])

        if lst[-1]==0:
            labels_test_all.append(-1)
            labels_test_male.append(-1)
        if lst[-1]==1:
            labels_test_all.append(1)
            labels_test_male.append(1)






    # Extracting features from female dataset
    for i in train_iter_female:
        lst = Data_female.iloc[i].values.tolist()
        audio_features_test_all.append(lst[0:12])
        audio_features_test_female.append(lst[0:12])

        text_features_test_all.append(lst[12:22])
        text_features_test_female.append(lst[12:22])

        vedio_features_test_all.append(lst[22:-2])
        vedio_features_test_female.append(lst[22:-2])

        if lst[-1]==0:
            labels_test_all.append(-1)
            labels_test_female.append(-1)
        if lst[-1]==1:
            labels_test_all.append(1)
            labels_test_female.append(1)



    # Finding accuracies for audio
    acc_test_audio_all.append(accuracy_score(labels_test_all,clf_audio.predict(audio_features_test_all))) # accuracy for all
    auc_test_audio_all.append(roc_auc_score(labels_test_all,np.array(clf_audio.predict_proba(audio_features_test_all))[:,0])) #auc audio all


    y_test_audio_male = clf_audio.predict(audio_features_test_male)  # male audio predicted classes
    y_test_audio_male_p = np.array(clf_audio.predict_proba(audio_features_test_male))[:,0]
    acc_test_audio_male.append(accuracy_score(labels_test_male,y_test_audio_male)) # male accuracey
    auc_test_audio_male.append(roc_auc_score(labels_test_male,np.array(clf_audio.predict_proba(audio_features_test_male))[:,0])) # auc audio male


    y_test_audio_female = clf_audio.predict(audio_features_test_female)  # male audio predicted classes
    y_test_audio_female_p = np.array(clf_audio.predict_proba(audio_features_test_female))[:,0]
    acc_test_audio_female.append(accuracy_score(labels_test_female,clf_audio.predict(audio_features_test_female))) # female accuracey
    auc_test_audio_female.append(roc_auc_score(labels_test_female,np.array(clf_audio.predict_proba(audio_features_test_female))[:,0])) #auc audio female


    # Finding accuracies for text
    acc_test_text_all.append(accuracy_score(labels_test_all,clf_text.predict(text_features_test_all))) # accuracy for all
    auc_test_text_all.append(roc_auc_score(labels_test_all,np.array(clf_text.predict_proba(text_features_test_all))[:,0])) #auc text all


    y_test_text_male = clf_text.predict(text_features_test_male)  # male audio predicted classes
    y_test_text_male_p = np.array(clf_text.predict_proba(text_features_test_male))[:,0]
    acc_test_text_male.append(accuracy_score(labels_test_male,clf_text.predict(text_features_test_male))) # male accuracey
    auc_test_text_male.append(roc_auc_score(labels_test_male,np.array(clf_text.predict_proba(text_features_test_male))[:,0])) #auc text male

    y_test_text_female = clf_text.predict(text_features_test_female)  # female audio predicted classes
    y_test_text_female_p = np.array(clf_text.predict_proba(text_features_test_female))[:,0]
    acc_test_text_female.append(accuracy_score(labels_test_female,clf_text.predict(text_features_test_female))) # female accuracey
    auc_test_text_female.append(roc_auc_score(labels_test_female,np.array(clf_text.predict_proba(text_features_test_female))[:,0])) #auc text female

    # Finding accuracies for vedio
    acc_test_vedio_all.append(accuracy_score(labels_test_all,clf_vedio.predict(vedio_features_test_all))) # accuracy for all
    auc_test_vedio_all.append(roc_auc_score(labels_test_all,np.array(clf_vedio.predict_proba(vedio_features_test_all))[:,0])) #auc vedio all


    y_test_vedio_male = clf_vedio.predict(vedio_features_test_male)  # male audio predicted classes
    y_test_vedio_male_p = np.array(clf_vedio.predict_proba(vedio_features_test_male))[:,0]
    acc_test_vedio_male.append(accuracy_score(labels_test_male,clf_vedio.predict(vedio_features_test_male))) # male accuracey
    auc_test_vedio_male.append(roc_auc_score(labels_test_male,np.array(clf_vedio.predict_proba(vedio_features_test_male))[:,0])) #auc vedio male


    y_test_vedio_female = clf_vedio.predict(vedio_features_test_female)  # male audio predicted classes
    y_test_vedio_female_p = np.array(clf_vedio.predict_proba(vedio_features_test_female))[:,0]
    acc_test_vedio_female.append(accuracy_score(labels_test_female,clf_vedio.predict(vedio_features_test_female))) # female accuracey
    auc_test_vedio_female.append(roc_auc_score(labels_test_female,np.array(clf_vedio.predict_proba(vedio_features_test_female))[:,0])) #auc vedio female

## Majoruty voting
    y_majority_female = y_test_text_female+y_test_audio_female+y_test_vedio_female
    y_majority_female [y_majority_female>0] = 1
    y_majority_female [y_majority_female<0] = -1
    acc_majority_female.append(accuracy_score(labels_test_female,y_majority_female))
    auc_majority_female.append(roc_auc_score(labels_test_female,y_majority_female))
    y_majority_male = y_test_text_male+y_test_audio_male+y_test_vedio_male
    y_majority_male [y_majority_male>0] = 1
    y_majority_male [y_majority_male<0] = -1
    acc_majority_male.append(accuracy_score(labels_test_male,y_majority_male))
    auc_majority_male.append(roc_auc_score(labels_test_male,y_majority_male))

## Average weights
    y_p_average_female = (w_audio*y_test_audio_female_p+w_text*y_test_text_female_p+w_vedio*y_test_vedio_female_p)/3.0 # finding the probabilities
    y_p_average_female = y_p_average_female.copy()
    y_p_average_female[y_p_average_female<.5] = 1
    y_p_average_female[y_p_average_female>=.5] = -1
    y_p_average_female_acc.append(accuracy_score(labels_test_female,y_p_average_female))
    y_p_average_female_auc.append(roc_auc_score(labels_test_female,y_p_average_female))

    y_p_average_male = (y_test_audio_male_p+y_test_text_male_p+y_test_vedio_male_p)/3.0 # finding the probabilities
    y_p_average_male = y_p_average_male.copy()
    y_p_average_male[y_p_average_male<.5] = 1
    y_p_average_male[y_p_average_male>=.5] = -1
    y_p_average_male_acc.append(accuracy_score(labels_test_male,y_p_average_male))
    y_p_average_male_auc.append(accuracy_score(labels_test_male,y_p_average_male))















print("mean accuracy female text---------------:",np.mean(acc_text_female))
print("mean auc female text---------------:",np.mean(auc_test_text_female))

print("mean accuracy male text---------------:",np.mean(acc_text_male))
print("mean auc male text---------------:",np.mean(auc_test_text_male))

print ("---------------")



print("mean accuracy female audio---------------:",np.mean(acc_audio_female))
print("mean auc female audio---------------:",np.mean(auc_test_audio_female))

print("mean accuracy male audio---------------:",np.mean(acc_audio_male))
print("mean auc male audio---------------:",np.mean(auc_test_audio_male))

print ("---------------")

print("mean accuracy female vedio---------------:",np.mean(acc_vedio_female))
print("mean auc female vedio---------------:",np.mean(auc_test_vedio_female))

print("mean accuracy male vedio---------------:",np.mean(acc_vedio_male))
print("mean auc female vedio---------------:",np.mean(auc_test_vedio_male))

print ("---------------")
print("mean accuracy Majority Voting female---------------:",np.mean(acc_majority_female))
print("mean auc Majority Voting female---------------:",np.mean(auc_majority_female))

print("mean accuracy Majority Voting male---------------:",np.mean(acc_majority_male))
print("mean auc Majority Voting male---------------:",np.mean(auc_majority_male))
print ("---------------")

print("mean accuracy Proposed weighting female---------------:",np.mean(y_p_average_female_acc))
print("mean auc Proposed weighting female---------------:",np.mean(y_p_average_female_auc))

print ("---------------")
print("mean accuracy Proposed weighting male---------------:",np.mean(y_p_average_male_acc))
print("mean auc Proposed weighting male---------------:",np.mean(y_p_average_male_auc))



