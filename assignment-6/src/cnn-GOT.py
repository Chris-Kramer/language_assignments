#!/usr/bin/env python3
"""
############### TO DO ############
* create readme file
    * Standard + something about results
* Make data processing to a function
* Add some parameters
    * Input and kernel size
    * learning rate (maybe even regularizers)
    * number of episodes a house are in series
    * number of words in dict
    * Dimensions and embedding matrix
    * epochs
    * Padding
    * (Look at the model, there are loads of places, that can be made to parameters)
* Make single qoutes to double
"""

"""
############### Import libs ###############
"""
# system tools
from pathlib import Path #I've started using Pathlib rather than os.path since it is much more convenient
import sys
import os
sys.path.append(os.path.join(".."))

# pandas, numpy, argparse
import argparse
import pandas as pd
import numpy as np
import re

# Homebrewed functions
import utils.classifier_utils as clf # Ross' classifier utility functions
import utils.cnn_utility as cnn # plotting and embedding matrix
import utils.preprocess_data as ppd #preprocesing data

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam #Our optimizers (Adam er meget god og hurtig)
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2 #Regularization
from tensorflow.keras.regularizers import L1 #Regularization
from tensorflow.keras.layers import Dropout

#Sklean
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# matplotlib
import matplotlib.pyplot as plt


def main():
    """
    ################ Parameters ############ 
    """
    #Create an argument parser from argparse
    ap = argparse.ArgumentParser(description = "[INFO] CNN for classifying houses in Game of Thrones")
    
    # minimum number of episodes a character from a house should appear
    ap.add_argument("-ne", "--n_episodes",
                    required = False,
                    default = 50,
                    type = int,
                    help = "[INFO] How many episodes should a member of the house at least appear in, [TYPE] int, [DEFAULT] 45")
    
    # Size of test data
    ap.add_argument("-ts", "--test_size",
                    required = False,
                    default = 0.2,
                    type = float,
                    help = "[INFO] The size of test data, [TYPE] float, [DEFAULT] 0.2")
     
    # Padding type
    ap.add_argument("-pa", "--padding",
                    required = False,
                    default = "post",
                    type = str,
                    help = "[INFO] Padding type, [TYPE] str, [DEFAULT] post")
    
    ap.add_argument("-nw", "--num_words",
                    required = False,
                    default = 10000,
                    type = int,
                    help = "[INFO] How many unique words the embedding dictionary should contain [TYPE] int, [DEFAULT] 10000")
    
    ap.add_argument("-em", "--embedding_dim",
                    required = False,
                    default = 100,
                    type = int,
                    help = "[INFO] How many embedding dimensions the embedding matrix should contain [TYPE] int [DEFAULT] 100")
    
    ap.add_argument("-pe", "--pretrained_embeddings",
                    required = False,
                    default = "../data/glove/glove.6B.100d.txt",
                    type = str,
                    help = "[INFO] The path to file with pretrained embeddings, [TYPE] str, [DEFAULT] ../data/glove/glove.6B.100d.txt")
    
    ap.add_argument("-l1", "--l1",
                    required = False,
                    default = 0.0001,
                    type = float,
                    help = "[INFO] The learning rate for L1 regularization (used in dense layer), [TYPE] float, [DEFAULT] 000.1")
    
    ap.add_argument("-l2", "--l2",
                    required = False,
                    default = 0.0001,
                    type = float,
                    help = "[INFO] The learning rate for L2 regularization (used in Conv1D layer), [TYPE] float, [DEFAULT] 000.1")
    
    ap.add_argument("-tr", "--trainable",
                    required = False,
                    default = True,
                    type = bool,
                    help = "[INFO] Should the embeddings be trainable, [TYPE] bool, [DEFAULT] = True")
    
    ap.add_argument("-fi", "--filters",
                    required = False,
                    default = [70, 30],
                    nargs = "*", # I wan't all passed arguments as one list
                    type = int,
                    help = "[INFO] How many filters should there be in the Conv1D and hidden dense layer, [TYPE] int, [DEFAUL] 70 30")
    
    ap.add_argument("-ks", "--kernel_size",
                    required = False,
                    default = 3,
                    type = int,
                    help = "[INFO] The size of the kernel in Conv1D layer, [TYPE] int, [DEFAULT] 3")
    
    ap.add_argument("-dr", "--dropout_rate",
                    required = False,
                    default = [0.2, 0,2],
                    nargs = "*",
                    type = float,
                    help = "[INFO] Dropout rate for first and second dropout layer, [TYPE] float, [DEFAULT] 0.2 0.2")
    
    ap.add_argument("-ep", "--epochs",
                    required = False,
                    default = 25,
                    type = int,
                    help = "[INFO] Amount of epochs the model should run, [TYPE] int, [DEFAULT] 25")

    ap.add_argument("-bs", "--batch_size",
                    required = False,
                    default = 10,
                    type = int,
                    help = "[INFO] The batch size for training and classification report, [TYPE] int, [DEFAULT] 10")
    #return parser
    args = vars(ap.parse_args())
        
    #save arguments in variables (for readability)
    n_episodes = args["n_episodes"]
    test_size = args["test_size"]
    num_words = args["num_words"]
    embedding_dim = args["embedding_dim"]
    pretrained_embeddings = args["pretrained_embeddings"]
    l1 = args["l1"]
    l2 = args["l2"]
    padding = args["padding"]
    trainable = args["trainable"]
    filters = args["filters"]
    kernel_size = args["kernel_size"]
    dropout_rate = args["dropout_rate"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
                    
    """
    ################ Load and preprocess data ################
    """
    print("preprocessing data ....")
    # Save X, y and label data
    X, y, label_names = ppd.get_xy_data(Path("../data/Game_of_Thrones_Script.csv"), n_episodes = n_episodes)
    
    # Save the length of the longest episode
    # Used to set maxlen of a doc
    highest_length = ppd.get_longest_entry(X)
    
    # Split data using sklearn
    X_train, X_test, y_train, y_test = train_test_split(X,  
                                                    y, 
                                                    test_size=test_size,
                                                    random_state=42) #For reproducibility
    
    # One hot key encoding
    #It looks like this performs better than sparse integer labels
    # The better performance might be completely random and might just be my own confirmation bias
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    """
    ################ Tokenize, padding, embedding ################
    """
    print("tokenizeing, padding and embedding data...")
    # ------- Tokenize --------
    # initialize tokenizer
    tokenizer = Tokenizer(num_words=num_words)
    # fit to training data 
    tokenizer.fit_on_texts(X_train)

    # tokenized training and test data
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)

    # overall vocabulary size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    
    # -------- Padding ---------
    # pad training data to maxlen
    X_train_pad = pad_sequences(X_train_toks, 
                                padding= padding, # Add padding at the end
                                maxlen=highest_length)
    # pad testing data to maxlen
    X_test_pad = pad_sequences(X_test_toks, 
                               padding= padding, #Add padding at the end
                               maxlen=highest_length)
    
    # ------- embedding --------
    #Create embedding matrix
    embedding_matrix = cnn.create_embedding_matrix(Path(pretrained_embeddings),
                                           tokenizer.word_index, 
                                           embedding_dim)
    """
    ################ Create model ################
    """
    print("creating model ....")
    # -------- Regularizers and model--------
    # I got better results when using a low learning rate
    l2 = L2(l2)
    l1 = L1(l1)
    model = Sequential() # Initialize model 
    
    # --------- Embedding layer-------
    model.add(Embedding(vocab_size,                  # vocab size from Tokenizer()
                        embedding_dim,               # embedding input layer size
                        weights=[embedding_matrix],  # pretrained embeddings
                        input_length=highest_length,         # maxlen of padded doc
                        trainable = trainable))
    
    # ------- Concolutional layer ------ 
    model.add(Conv1D(filters[0], kernel_size,
                    activation = "relu", #Relu activation function
                    kernel_regularizer=l2))# L2 regularization
    
    # ------ Dropout layer ------
    model.add(Dropout(dropout_rate[0]))
    
    # ------ Pooling layer ------
    model.add(GlobalMaxPool1D())
    
    # ----- Dense (hidden) activation layer ------
    model.add(Dense(filters[1], activation="relu", kernel_regularizer=l1))#L1 regularization
    
    # ------ Dropout layer ------
    model.add(Dropout(dropout_rate[1]))
    
    # ------ Output layer -------
    model.add(Dense(len(label_names), activation="softmax"))

    # ------ compile model -----
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    
    """
    ################ Summarize and evaluate model ################
    """
    print("Summarizing model ....")
    #-------- Summarize model -------
    model.summary() # print summary
    #plot model architecture
    plot_model(model, to_file = Path("../output/model_architecture.png"), show_shapes=True, show_layer_names=True)
    
    #------- Train model on 20 epochs -------
    print("Training model ....")
    history = model.fit(X_train_pad, y_train,
                        epochs=epochs,
                        verbose=True,
                        validation_data=(X_test_pad, y_test),
                        batch_size=batch_size)
    
    #----- evaluate model and plot ------- 
    print("evaluating model ....")
    loss, accuracy = model.evaluate(X_train_pad, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    cnn.plot_history(history, epochs = epochs, output = Path("../output/performance_cnn.png")) # plot performance
    
    predictions = model.predict(X_test_pad, batch_size = batch_size)
    print(classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=label_names))
#Define behaviour when called from terminal
if __name__ == "__main__":
    main()