#!/usr/bin/env python3
"""
############### Import libs ###############
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
    ap = argparse.ArgumentParser(description = "[INFO] CNN for classifying houses in Game of Thrones",
                                formatter_class = RawTextHelpFormatter)
    
    # minimum number of episodes a character from a house should appear
    ap.add_argument("-ne", "--n_episodes",
                    required = False,
                    default = 50,
                    type = int,
                    help =
                    "[INFO] How many episodes should a house at least appear in \n"
                    "[INFO] There are 72 episodes in total \n"
                    "[TYPE] int \n" 
                    "[DEFAULT] 50"
                    "[EXAMPLE] --n_episodes 60")
    
    # Size of test data
    ap.add_argument("-ts", "--test_size",
                    required = False,
                    default = 0.2,
                    type = float,
                    help =
                    "[INFO] The size of test data as a percentage \n" 
                    "[INFO] The training size will be adjusted automatically \n"
                    "[TYPE] float \n"
                    "[DEFAULT] 0.2 \n"
                    "[EXAMPLE] --test_size 0.1")
     
    # Padding type
    ap.add_argument("-pa", "--padding",
                    required = False,
                    default = "post",
                    type = str,
                    help =
                    "[INFO] Padding type \n"
                    "[TYPE] str \n"
                    "[DEFAULT] post \n"
                    "[EXAMPLE] --padding pre")
    
    # Number of words in embedding dict
    ap.add_argument("-nw", "--num_words",
                    required = False,
                    default = 10000,
                    type = int,
                    help =
                    "[INFO] How many unique words the embedding dictionary should contain \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 10000 \n"
                    "[EXAMPLE] --num_words 8000")
    
    # Embedding dimensions
    ap.add_argument("-em", "--embedding_dim",
                    required = False,
                    default = 100,
                    type = int,
                    help = 
                    "[INFO] How many embedding dimensions the embedding matrix should contain \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 100 \n"
                    "[EXAMPLE] --embedding_dim 50")
    
    #Pretrained embeddings
    ap.add_argument("-pe", "--pretrained_embeddings",
                    required = False,
                    default = "glove.6B.100d.txt",
                    type = str,
                    help = 
                    "[INFO] The Glove pretrained embeddings \n"
                    "[INFO] Must be located in the folder '../data/glove' \n"
                    "[TYPE] str \n"
                    "[DEFAULT] glove.6B.100d.txt \n"
                    "[EXAMPLE] --pretrained_embeddings glove.6B.50d.txt")
    
    # l1 regularization
    ap.add_argument("-l1", "--l1",
                    required = False,
                    default = 0.0001,
                    type = float,
                    help =
                    "[INFO] The learning rate for L1 regularization (used in dense layer) \n"
                    "[TYPE] float \n"
                    "[DEFAULT] 0.0001 \n"
                    "[EXAMPLE] --l1 0.001")
    
    #l2 regularization
    ap.add_argument("-l2", "--l2",
                    required = False,
                    default = 0.0001,
                    type = float,
                    help =
                    "[INFO] The learning rate for L2 regularization (used in Conv1D layer) \n"
                    "[TYPE] float \n"
                    "[DEFAULT] 0.0001 \n"
                    "[EXAMPLe] --l2 0.001")
    
    # Trainable parameters or not
    ap.add_argument("-tr", "--trainable",
                    required = False,
                    default = True,
                    type = bool,
                    help =
                    "[INFO] Should the embeddings be trainable \n"
                    "[INFO] Must be either 'True' or 'False'"
                    "[TYPE] bool \n"
                    "[DEFAULT] True \n"
                    "[EXAMPLE] --trainable False")
    
    #filters in conv1d and dense
    ap.add_argument("-fi", "--filters",
                    required = False,
                    default = [70, 30],
                    nargs = "*", # I wan't all passed arguments as one list
                    type = int,
                    help =
                    "[INFO] How many filters should there be in the Conv1D and hidden dense layer \n"
                    "[TYPE] list of ints \n"
                    "[DEFAUL] 70 30 \n"
                    "[EXAMPLE] --filters 60 20")
    
    #kernel seize in conv1d
    ap.add_argument("-ks", "--kernel_size",
                    required = False,
                    default = 3,
                    type = int,
                    help =
                    "[INFO] The size of the kernel in Conv1D layer \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 3 \n"
                    "[EXAMPLE] --kernel_size 5")
    
    #dropout rates for both dropout layers
    ap.add_argument("-dr", "--dropout_rate",
                    required = False,
                    default = [0.2, 0,2],
                    nargs = "*",
                    type = float,
                    help = 
                    "[INFO] Dropout rate for first and second dropout layer"
                    "[TYPE] list of floats \n"
                    "[DEFAULT] 0.2 0.2"
                    "[EXAMPLE] --dropout_rate 0.1 0.1")
    
    #epochs
    ap.add_argument("-ep", "--epochs",
                    required = False,
                    default = 25,
                    type = int,
                    help =
                    "[INFO] Amount of epochs the model should run \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 25 \n"
                    "[EXAMPLE] --epochs 10")
    
    #batch size
    ap.add_argument("-bs", "--batch_size",
                    required = False,
                    default = 10,
                    type = int,
                    help =
                    "[INFO] The batch size for training and evaluation \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 10 \n"
                    "[EXAMPLE] --batch_size 32")
    #return parser
    args = vars(ap.parse_args())
        
    #save arguments in variables (for readability)
    n_episodes = args["n_episodes"]
    test_size = args["test_size"]
    num_words = args["num_words"]
    embedding_dim = args["embedding_dim"]
    pretrained_embeddings = os.path.join("..", "data", "glove", args["pretrained_embeddings"])
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
    file_path = os.path.join("..", "data", "Game_of_Thrones_Script.csv")
    # Save X, y and label data
    X, y, label_names = ppd.get_xy_data(file_path, n_episodes = n_episodes)
    
    # Save the length of the longest episode
    # Use to set maxlen of a doc
    highest_length = ppd.get_longest_entry(X)
    
    # Split data using sklearn
    X_train, X_test, y_train, y_test = train_test_split(X,  
                                                    y, 
                                                    test_size=test_size,
                                                    random_state=42) #For reproducibility
    
    # One hot key encoding
    # It looks like this performs better than sparse integer labels
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
    embedding_matrix = cnn.create_embedding_matrix(pretrained_embeddings,
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
    file_path = os.path.join("..", "output", "model_architecture.png")
    plot_model(model, to_file = file_path, show_shapes=True, show_layer_names=True)
    
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
    file_path = os.path.join("..", "output", "performance_cnn.png")
    cnn.plot_history(history, epochs = epochs, output = file_path) # plot performance
    
    predictions = model.predict(X_test_pad, batch_size = batch_size)
    print(classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=label_names))
    
#Define behaviour when called from terminal
if __name__ == "__main__":
    main()