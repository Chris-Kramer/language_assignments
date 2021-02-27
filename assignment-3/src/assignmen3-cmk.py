#!/usr/bin/env python
"""----------- Import libs----------
"""
import os
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from spacytextblob.spacytextblob import SpacyTextBlob

"""---------- Initialise spacy, Textblob and add pipe--------------
"""
# Initialise spaCy
nlp = spacy.load("en_core_web_sm")
#Create textblob
spacy_text_blob = SpacyTextBlob()
#Add pipe
nlp.add_pipe(spacy_text_blob)

"""---------- Parameters----------
"""
# Read csv and save in variable
data = pd.read_csv(os.path.join("..", "data", "abcnews-date-text.csv"))
#Get a list of every date
dates = data["publish_date"].unique()

def main():

    """---------- Calculate average sentiments----------
    """
    #List of  mean score for each day
    mean_scores = []
    
    day_index = 1 #for printing to terminal
    length_dates = len(dates) #Used for printing to terminal
    #For each day
    for day in dates:
        print(f"Day: {day_index} / {length_dates}")
        #This variable will contain all sentiment scores for this day
        scores = []
        #This is a list of every headline in the day we are looping through
        headlines = data[data["publish_date"]==day]
    
        #For each headline from that day
        for headline in nlp.pipe(headlines["headline_text"], batch_size=500):
            #Calculate sentiment score for headline
            headline_score = headline._.sentiment.polarity
            #Append headline score
            scores.append(headline_score)
            
        #calculate mean score for the day
        mean_scores.append(np.mean(scores))
        #increase day index
        day_index += 1
    """---------- Calculate rolling averages----------
    """
    #Calculate 7 day rolling average
    smoothed_sentiment_weeks = pd.Series(mean_scores).rolling(7).mean()
    #Calculate 30 day rolling average
    smoothed_sentiment_months = pd.Series(mean_scores).rolling(30).mean()
    #Calculate 365 day rolling average
    smoothed_sentiment_years = pd.Series(mean_scores).rolling(365).mean()

    """ ---------- Create figure ----------
    """

    #Create figure
    fig = plt.figure(figsize = (15,10)) 
    #Create a grid for readability
    plt.grid()

    #Plot average sentiment for every day
    plt.plot(mean_scores)
    #Plot average sentiment for 7 days rolling
    plt.plot(smoothed_sentiment_weeks)
    #Plot average sentiment for 30 days rolling
    plt.plot(smoothed_sentiment_months)
    #Plot average sentiment for 365 days rolling
    plt.plot(smoothed_sentiment_years)

    #Create title, labels and legend
    plt.title("Headline sentiment since 2003",fontsize= 20)
    plt.xlabel("Years since 2003", fontsize= 15, labelpad=10)
    plt.ylabel("Sentiment Score", fontsize= 15)
    plt.legend(["Average daily sentiment", "Weekly rolling average", "Monthly rolling average", "Yearly rolling average"],
               loc='upper right',
               fontsize= 12)

    #Set x ticks to be in years rather than default (days)
    plt.xticks(np.arange(0, len(mean_scores)+1,365), range(0,18))
    
    #Show plot and save figure
    fig.savefig("../output/sentiment_score.jpg")
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()