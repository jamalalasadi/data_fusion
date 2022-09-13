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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
import sys
from sklearn.metrics import roc_auc_score


def conf_mat_param(pred_labels,true_labels):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == -1, true_labels == -1))  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == -1))  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == -1, true_labels == 1))
    return (TP,FP,TN,FN)


ratio = 0.60  # the training testing ratio

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

# all accuracy variables
acc_audio_all = []
acc_text_all = []
acc_vedio_all = []
acc_test_audio_all = []
acc_test_text_all = []
acc_test_vedio_all = []


# all male confusion matrix variables
cm_audio_male = []
cm_text_male = []
cm_vedio_male = []
cm_test_audio_male = []
cm_test_text_male = []
cm_test_vedio_male = []
cm_majority_male = []
cm_average_male = []



# all female confusion matrix variables
cm_audio_female = []
cm_text_female = []
cm_vedio_female = []
cm_test_audio_female = []
cm_test_text_female = []
cm_test_vedio_female = []
cm_majority_female = []
cm_average_female = []

# Area under curve variables female

auc_test_audio_female = []
auc_test_text_female = []
auc_test_vedio_female = []
auc_majority_female = []
auc_average_female = []

# Area under curve variables male

auc_test_audio_male = []
auc_test_text_male = []
auc_test_vedio_male = []
auc_majority_male = []
auc_average_male = []

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
            train_labels_all.append(-1) # nonbully
            lable_train_female.append(-1)
        if lst[-1]==1:
            train_labels_all.append(1) #bully
            lable_train_female.append(1)


    # Training and find accuracies for audio features
    clf_audio =  GaussianNB().fit(audio_features_all,train_labels_all)# Classifier training for audio

    acc_audio_all.append(accuracy_score(train_labels_all,clf_audio.predict(audio_features_all))) # accuracy for all

    acc_audio_male.append(accuracy_score(lable_train_male,clf_audio.predict(audio_features_male))) # male accuracey

    acc_audio_female.append(accuracy_score(lable_train_female,clf_audio.predict(audio_features_female))) # female accuracey

    w_audio = acc_audio_all[-1]-(acc_audio_male[-1]-acc_audio_female[-1])
    # w_audio = 0.2*acc_audio_all[-1]
    # Training and finding accuracies for text features
    clf_text =  GaussianNB().fit(text_features_all,train_labels_all) # Calssifier training for text

    acc_text_all.append(accuracy_score(train_labels_all,clf_text.predict(text_features_all))) # accuracy for all

    acc_text_male.append(accuracy_score(lable_train_male,clf_text.predict(text_features_male))) # male accuracey

    acc_text_female.append(accuracy_score(lable_train_female,clf_text.predict(text_features_female))) # female accuracey

    w_text = acc_text_all[-1]-(acc_text_male[-1]-acc_text_female[-1])
    #w_text = 0.7*acc_text_all[-1]


    # Training and finding accuracies for vedio features

    clf_vedio =  GaussianNB().fit(vedio_features_all,train_labels_all)# Classifier training for vedio

    acc_vedio_all.append(accuracy_score(train_labels_all,clf_vedio.predict(vedio_features_all))) # accuracy for all

    acc_vedio_male.append( accuracy_score(lable_train_male,clf_vedio.predict(vedio_features_male))) # male accuracy

    acc_vedio_female.append(accuracy_score(lable_train_female,clf_vedio.predict(vedio_features_female))) # female accuracey

    w_vedio = 0.55*acc_vedio_all[-1]-(acc_vedio_male[-1]-acc_vedio_female[-1])
    #w_vedio = 0.35*acc_vedio_all[-1]
    

    # normalize the weights
    s = w_audio+w_text+w_vedio
    w_audio = w_audio/s
    w_text = w_text/s
    w_vedio = w_vedio/s

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

    # Finding accuracies and confusion matrix for audio
    acc_test_audio_all.append(accuracy_score(labels_test_all,clf_audio.predict(audio_features_test_all))) # accuracy for all5

    y_test_audio_male = clf_audio.predict(audio_features_test_male)  # male audio predicted classes
    y_test_audio_male_p = np.array(clf_audio.predict_proba(audio_features_test_male))[:,1]
    acc_test_audio_male.append(accuracy_score(labels_test_male,y_test_audio_male)) # male accuracey


    y_test_audio_female = clf_audio.predict(audio_features_test_female)  # male audio predicted classes
    y_test_audio_female_p = np.array(clf_audio.predict_proba(audio_features_test_female))[:,1]
    acc_test_audio_female.append(accuracy_score(labels_test_female,clf_audio.predict(audio_features_test_female))) # female accuracey
    cm_audio_female.append(mt.confusion_matrix(labels_test_female,y_test_audio_female))
    cm_audio_male.append(mt.confusion_matrix(labels_test_male,y_test_audio_male))
    
    auc_test_audio_female.append(roc_auc_score(labels_test_female,y_test_audio_female_p)) # auc audio male
    auc_test_audio_male.append(roc_auc_score(labels_test_male,y_test_audio_male_p)) # auc audio male

       
    # print ('------------------------')
    # 
    # 
   
    # print ('------------------------')

    # Finding accuracies for text
    acc_test_text_all.append(accuracy_score(labels_test_all,clf_text.predict(text_features_test_all))) # accuracy for all

    y_test_text_male = clf_text.predict(text_features_test_male)  # male audio predicted classes
    y_test_text_male_p = np.array(clf_text.predict_proba(text_features_test_male))[:,1]
    acc_test_text_male.append(accuracy_score(labels_test_male,y_test_text_male)) # male accuracey

    y_test_text_female = clf_text.predict(text_features_test_female)  # female audio predicted classes
    y_test_text_female_p = np.array(clf_text.predict_proba(text_features_test_female))[:,1]
    acc_test_text_female.append(accuracy_score(labels_test_female,clf_text.predict(text_features_test_female))) # female accuracey
    
    cm_text_female.append(mt.confusion_matrix(labels_test_female,y_test_text_female))
    cm_text_male.append(mt.confusion_matrix(labels_test_male,y_test_text_male))
    
    auc_test_text_female.append(roc_auc_score(labels_test_female,y_test_text_female_p)) # auc audio male
    auc_test_text_male.append(roc_auc_score(labels_test_male,y_test_text_male_p)) # auc audio male

    # Finding accuracies for vedio
    acc_test_vedio_all.append(accuracy_score(labels_test_all,clf_vedio.predict(vedio_features_test_all))) # accuracy for all

    y_test_vedio_male = clf_vedio.predict(vedio_features_test_male)  # male audio predicted classes
    y_test_vedio_male_p = np.array(clf_vedio.predict_proba(vedio_features_test_male))[:,1]
    acc_test_vedio_male.append(accuracy_score(labels_test_male,clf_vedio.predict(vedio_features_test_male))) # male accuracey


    y_test_vedio_female = clf_vedio.predict(vedio_features_test_female)  # male audio predicted classes
    y_test_vedio_female_p = np.array(clf_vedio.predict_proba(vedio_features_test_female))[:,1]
    acc_test_vedio_female.append(accuracy_score(labels_test_female,clf_vedio.predict(vedio_features_test_female))) # female accuracey
    
    cm_vedio_female.append(mt.confusion_matrix(labels_test_female,y_test_vedio_female))
    cm_vedio_male.append(mt.confusion_matrix(labels_test_male,y_test_vedio_male))
    
    auc_test_vedio_female.append(roc_auc_score(labels_test_female,y_test_vedio_female_p)) # auc audio male
    auc_test_vedio_male.append(roc_auc_score(labels_test_male,y_test_vedio_male_p)) # auc audio male

## Majoruty voting
    y_majority_female = y_test_text_female+y_test_audio_female+y_test_vedio_female
    y_majority_female [y_majority_female>0] = 1
    y_majority_female [y_majority_female<0] = -1
    acc_majority_female.append(accuracy_score(labels_test_female,y_majority_female))
    cm_majority_female.append(mt.confusion_matrix(labels_test_female,y_majority_female))

    y_majority_male = y_test_text_male+y_test_audio_male+y_test_vedio_male
    y_majority_male [y_majority_male>0] = 1
    y_majority_male [y_majority_male<0] = -1
    acc_majority_male.append(accuracy_score(labels_test_male,y_majority_male))
    cm_majority_male.append(mt.confusion_matrix(labels_test_male,y_majority_male))
    
    auc_majority_female.append(roc_auc_score(labels_test_female,y_majority_female))
    auc_majority_male.append(roc_auc_score(labels_test_male,y_majority_male))

    
## Average weights
    y_p_average_female = (w_audio*y_test_audio_female_p+w_text*y_test_text_female_p+w_vedio*y_test_vedio_female_p) # finding the probabilities
    y_p_average_female_c = y_p_average_female.copy()
    y_p_average_female[y_p_average_female_c>.5] = 1
    y_p_average_female[y_p_average_female_c<=.5] = -1
    y_p_average_female_acc.append(accuracy_score(labels_test_female,y_p_average_female))
    cm_average_female.append(mt.confusion_matrix(labels_test_female,y_p_average_female))

    y_p_average_male = (w_audio*y_test_audio_male_p+w_text*y_test_text_male_p+w_vedio*y_test_vedio_male_p)# finding the probabilities
    y_p_average_male_c = y_p_average_male.copy()
    y_p_average_male[y_p_average_male_c>.5] = 1
    y_p_average_male[y_p_average_male_c<=.5] = -1
    y_p_average_male_acc.append(accuracy_score(labels_test_male,y_p_average_male))
    cm_average_male.append(mt.confusion_matrix(labels_test_male,y_p_average_male))
    
    auc_average_male.append(roc_auc_score(labels_test_male,y_p_average_male))
    auc_average_female.append(roc_auc_score(labels_test_female,y_p_average_female))

# print("mean accuracy male all---------------:",np.mean( y_p_average_male_acc))
# print("mean accuracy female all---------------:",np.mean( y_p_average_female_acc))



print("mean accuracy female text---------------:",np.mean(acc_text_female))
print("mean accuracy male text---------------:",np.mean(acc_text_male))
print("mean confusion matrix female text-------:",np.mean(cm_text_female,0))
print("mean confusion matrix male text-------:",np.mean(cm_text_male,0))
print("mean auc female text------",np.mean(auc_test_text_female))
print("mean auc male text------",np.mean(auc_test_text_male))


print ("---------------")
print("mean accuracy female audio---------------:",np.mean(acc_audio_female))
print("mean accuracy male audio---------------:",np.mean(acc_audio_male))
print("mean confusion matrix female audio-------:",np.mean(cm_audio_female,0))
print("mean confusion matrix male audio-------:",np.mean(cm_audio_male,0))
print("mean auc female audio------",np.mean(auc_test_audio_female))
print("mean auc male audio------",np.mean(auc_test_audio_male))
print ("---------------")
print("mean accuracy female vedio---------------:",np.mean(acc_vedio_female))
print("mean accuracy male vedio---------------:",np.mean(acc_vedio_male))
print("mean confusion matrix female vedio-------:",np.mean(cm_vedio_female,0))
print("mean confusion matrix male vedio-------:",np.mean(cm_vedio_male,0))
print("mean auc female vedio------",np.mean(auc_test_vedio_female))
print("mean auc male vedio------",np.mean(auc_test_vedio_male))
print ("---------------")
print("mean accuracy Majority Voting female---------------:",np.mean(acc_majority_female))
print("mean accuracy Majority Voting male---------------:",np.mean(acc_majority_male))
print("mean confusion matrix Majority Voting female-------:",np.mean(cm_majority_female,0))
print("mean confusion matrix Majority Voting male-------:",np.mean(cm_majority_male,0))
print("mean auc female Majority Voting------",np.mean(auc_majority_female))
print("mean auc male Majority Voting------",np.mean(auc_majority_male))
print ("---------------")
print("mean accuracy Proposed weighting female---------------:",np.mean(y_p_average_female_acc))
print("mean accuracy Proposed weighting male---------------:",np.mean(y_p_average_male_acc))
print("mean confusion matrix Proposed weighting female-------:",np.mean(cm_average_female,0))
print("mean confusion matrix Proposed weighting male-------:",np.mean(cm_average_male,0))
print("mean auc female Proposed ------",np.mean(auc_average_female))
print("mean auc male Proposed------",np.mean(auc_average_male))


# finding average confusion matrix




