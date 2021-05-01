#!/usr/bin/env python3
"""
############# Import libs #############
"""
# system tools
import sys
import os
sys.path.append(os.path.join(".."))

# pandas, numpy, argparse
import argparse
from argparse import RawTextHelpFormatter # Formatting -help
import pandas as pd
import numpy as np
import re

# Homebrewed functions
import utils.classifier_utils as clf # import Ross' classifier utility functions
import utils.cnn_utility as cnn # plotting and embedding matrix
import utils.preprocess_data as ppd #preprocesing data

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.metrics import classification_report

# matplotlib
import matplotlib.pyplot as plt

def main():
    """
    ################ Parameters ############ 
    """
    #Create an argument parser from argparse
    ap = argparse.ArgumentParser(description = "[INFO] Logistic regression for classifying houses in Game of Thrones",
                                formatter_class = RawTextHelpFormatter)
    
    # minimum number of episodes a character from a house should appear
    ap.add_argument("-ne", "--n_episodes",
                    required = False,
                    default = 50,
                    type = int,
                    help =
                    "[INFO] How many episodes should a member of the house at least appear in \n"
                    "[INFO] There are 72 episodes in total \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 50 \n"
                    "[EXAMPLE] --n_episodes 60")
    
    # Size of test data
    ap.add_argument("-ts", "--test_size",
                    required = False,
                    default = 0.2,
                    type = float,
                    help =
                    "[INFO] The size of test data \n"
                    "[INFO] The training size will be adjusted automatically \n"
                    "[TYPE] float \n"
                    "[DEFAULT] 0.2 \n"
                    "[EXAMPLE] --test_size 0.1")
    
    # n_splits for cross validation
    ap.add_argument("-ns", "--n_splits",
                    required = False,
                    default = 50,
                    type = int,
                    help =
                    "[INFO] Amounts of shufflesplits during the cross validation \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 50")
    #return parser
    args = vars(ap.parse_args())
    
    #Save in variables (for readability) 
    n_episodes = args["n_episodes"]
    test_size = args["test_size"]
    n_splits = args["n_splits"]
    
    """
    ################ Load and preprocess data ################
    """
    print("preprocessing data ....")
    # Save X, y and label data
    file_path = os.path.join("..", "data", "Game_of_Thrones_Script.csv")
    X, y, label_names = ppd.get_xy_data(file_path, n_episodes = n_episodes)
    
    # Split data using sklearn
    X_train, X_test, y_train, y_test = train_test_split(X,  
                                                    y, 
                                                    test_size=test_size,
                                                    random_state=42) #For reproducibility
    """
    ############## Vectorize ############
    """
    print("vectorizing data ...")
    vectorizer = CountVectorizer() # Counting word distributions
    # vectorize training data
    X_train_feats = vectorizer.fit_transform(X_train)
    # # vectorize test data
    X_test_feats = vectorizer.transform(X_test)
    
    """
    ########## create and plot model #############
    """
    print("creating and plotting model...")
    #Create model
    classifier = LogisticRegression(random_state=42, max_iter = 1000).fit(X_train_feats, y_train)
    #Create predictions for y data
    y_pred = classifier.predict(X_test_feats)
    # Create classification report
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    print(classifier_metrics)
    #plot confusion matrix
    file_path = os.path.join("..", "output", "linear_regression_confusion_matrix.png")
    clf.plot_cm(y_test, y_pred, normalized=True, output = file_path)
    
    """
    ########## Cross validation ##########
    """
    print("cross validating model ...")
    # Vectorize full dataset
    X_vect = vectorizer.fit_transform(X)
    
    # initialise cross-validation method
    title = "Learning Curves (Logistic Regression)"
    cv = ShuffleSplit(n_splits = n_splits, test_size = test_size, random_state=42) #Cross validation model
    
    # cross validate on data
    model = LogisticRegression(random_state=42, max_iter = 1000) #logistic regresion model
    # plot and cross validate
    file_path = os.path.join("..", "output", "cross_validation_logistic_regression.png")
    clf.plot_learning_curve(model, title, X_vect, y,
                            cv=cv, n_jobs=4,
                            output = file_path)

if __name__ == "__main__":
    main()