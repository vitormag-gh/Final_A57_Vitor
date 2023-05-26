# -*- coding: utf-8 -*-
"""
Created on Thu May 25 21:46:01 2023

@author: riskf
"""

import os
import warnings
import sys

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics_train(y_train, y_naive_bayes):
    accuracy = metrics.accuracy_score(y_train, y_naive_bayes)
    recall_score = metrics.recall_score(y_train, y_naive_bayes)
    f1_score =  metrics.f1_score(y_train, y_naive_bayes)
    return accuracy, recall_score, f1_score

def eval_metrics_test(X_test, y_test,y_naive_bayes):
    recall_score  = metrics.recall_score(y_test, y_naive_bayes)
    f1_score      = metrics.f1_score(y_test, y_naive_bayes)
    return recall_score, f1_score


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    _random_state=44
    # OBTENIR LE DATASET
    file='pointure.data'
    try:
        df = pd.read_csv(file)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e)

    # PRE-TRAITEMENT DES DONNÉES
    label_encoder = preprocessing.LabelEncoder()
    input_classes = ['masculin','féminin']
    label_encoder.fit(input_classes)

    # transformer un ensemble de classes
    encoded_labels = label_encoder.transform(df['Genre'])
    df['Genre'] = encoded_labels
    
    # MATRICE DE CORRELATION ET DE PERSON
    corr = df.corr().round(3)

    # SEPARER LE DATASET EN TRAIN ET TEST
    X = df.iloc[:, lambda df: [1, 2, 3]]
    y = df.iloc[:, 0]
    
    #decomposer les donnees predicteurs en training/testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=_random_state)
    
    # FAIRE APPRENDRE LE MODELE
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
        
    # EVALUATION SUR LE TRAIN
    y_naive_bayes1 = gnb.predict(X_train)
    accuracy, recall_score, f1_score=eval_metrics_train(y_train, y_naive_bayes1)
    
    # EVALUATION SUR LE TEST
    y_naive_bayes2 = gnb.predict(X_test)
    recall_score_test, f1_score_test =eval_metrics_test(X_test,y_test, y_naive_bayes2)
    
    
    with open("metrics.txt", 'w') as outfile:
            outfile.write("Train_Accuracy:  {0:2.1f} \n".format(accuracy))
            outfile.write("Train_Recall_score: {0:2.1f}\n".format(recall_score))
            outfile.write("Train_F1_Score: {0:2.1f}\n".format(f1_score))
            outfile.write("Test_Recall_Score: {0:2.1f}\n".format(recall_score_test))
            outfile.write("Test_F1_Score: {0:2.1f}\n".format(f1_score_test))
            
            