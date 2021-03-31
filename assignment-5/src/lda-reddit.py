#!/usr/bin/env python
"""
---------- Import libs -----------
"""
# standard libraries
import sys,os
#sys.path.append(os.getcwd())
sys.path.append(os.path.join(".."))
from pprint import pprint
import datetime
import numpy as np
import argparse

# data and nlp
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.max_length = 3951751

# visualisation
import pyLDAvis.gensim
import seaborn as sns
from matplotlib import rcParams
from matplotlib import pyplot as plt

# LDA tools
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from utils import lda_utils

# warnings
import logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

def main():
    """
    ------------ parameters ------------
    """
    #Create an argument parser from argparse
    ap = argparse.ArgumentParser(description = "[INFO] Topic modelling of r/WallStreetBets")
    
    #start date
    ap.add_argument("-sd", "--start_date",
                    required = False,
                    default = [2021, 1, 10],
                    nargs = "*",
                    help = "The start date for reddit posts (YYYY-D-M). DEFAULT = 2021 1 10")
    
    #end date
    ap.add_argument("-ed", "--end_date",
                    required = False,
                    default = [2021, 3, 1],
                    nargs = "*",
                    help = "The end date for reddit posts (YYYY-D-M). DEFAULT = 2021 3 1")
    
    #amount of topics 
    ap.add_argument("-t", "--topics",
                    required = False,
                    default = 3,
                    type = int,
                    help = "The amount of topics. DEFAULT = 3")

  
    #Create save arguments in a variable
    args = vars(ap.parse_args())
    
    #Save in variables for readability
    start_date = args["start_date"]
    end_date = args["end_date"]
    topics = args["topics"]
    """
    ---------- Read and clean data-----------
    """
    #Read data
    data = pd.read_csv("../data/r_wallstreetbets_posts.csv")
    #Get a subset of the dataset
    data = data[["title", "created_utc"]]
    #Make created utc into a datetime format
    data["created_utc"] = pd.to_datetime(data["created_utc"], unit = "s").dt.date
    #Choose dates after provided date
    data = data[data["created_utc"] > datetime.date(int(start_date[0]), int(start_date[1]), int(start_date[2]))]
    #Chose dates before provided date
    data = data[data["created_utc"] < datetime.date(int(end_date[0]), int(end_date[1]), int(end_date[2]))]
    #Sort the data according to dates
    data = data.sort_values("created_utc")
    
    """
    ----------- Group posts according to date ------------
    """
    #This will contain all posts for each month in a list of concactinated strings
    dates = []
    #For every unique date
    for date in data["created_utc"].unique():
        #Get the posts from that day
        posts = data[data["created_utc"]==date]["title"]
        #concactinate and append the posts
        dates.append(" ".join([str(post) for post in posts]))
        
    """
    ---------- Preprocess data ------------
    """
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(dates, min_count=3, threshold=100) 
    trigram = gensim.models.Phrases(bigram[dates], threshold=100)

    #Fit the models to the data
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    #Process data
    data_processed = lda_utils.process_words(dates, #We are using the chunks
                                             nlp, #We are using our nlp 
                                             bigram_mod, #We fit it to our bigrams
                                             trigram_mod, #We fit it to our trigrams
                                             allowed_postags=["NOUN"]) #We are only finding nouns
    # Create Dictionary
    id2word = corpora.Dictionary(data_processed)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_processed]
    
    """
    ------------ Test and build model ----------
    """
    # Can take a long time to run.
    #I'm starting by calculating the coherence values and perplexity, that way, I can better determine the appropriate amount of topics 
    model_list, coherence_values = lda_utils.compute_coherence_values(texts=data_processed,
                                                                      corpus=corpus, 
                                                                      dictionary=id2word,  
                                                                      start=2, 
                                                                      limit=10,  
                                                                      step=2)
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus, #Our vectorized corpus - list of lists of tupples
                                           id2word=id2word, #Our gensim dictionary (mapping words to IDs)
                                           num_topics=topics, #The number of topics
                                           random_state=100, #Let's create a random state to make the results reproducible
                                           chunksize=150,  # Let's set the chunksize to 150, batch data for efficiency
                                           passes=10, #Is the same as epochs, how many times do we wanne go trhough the data
                                           iterations=100, # How often are we going over a single document. (Related to passes)
                                           per_word_topics=True,  #define word distributions 
                                           minimum_probability=0.0) #Minimum value. Include topics with zero probability
    
    # Compute Perplexity and print it to terminal
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                     texts=data_processed, 
                                     dictionary=id2word, 
                                     coherence='c_v')
    
    #Print coherence the higher the better
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    
    #Print topics
    pprint(lda_model.print_topics())
    
    #Create interactive board of topics
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    # Save the board as a html-file
    pyLDAvis.save_html(vis, "../output/lda-board_wallStreetBets.html")
    
    """
    ---------- Create dataframe and plot topics ----------
    """
    # Get topics from the corpus
    values = list(lda_model.get_document_topics(corpus))
    #Create list which can be splitted to a matrix
    split = []
    #For each list of topics in the documents
    for entry in values:
        #Create list for the topics prevelance
        topic_prevelance = []
        #For each topic in the list of topics
        for topic in entry:
            #Append the topics prevalence 
            topic_prevelance.append(topic[1])
        #append list of topic prevalences 
        split.append(topic_prevelance)
        
    #This is ugly as f**k, but this is the best solution I could come up with to plot dates
    #Split dataframe to wide format
    df = pd.DataFrame(map(list,zip(*split)))
    #make it long format
    df = df.transpose()
    #add column with dates
    df["date"] = data["created_utc"].unique()
    #Set dates to be index
    df = df.set_index("date")
    

    #plot lineplot
    lineplot = sns.relplot(data=df.rolling(5).mean(), kind = "line", legend = True)
    #Add title to legend
    lineplot._legend.set_title("Topics")
    #Set size
    lineplot.fig.set_size_inches(20,10)
    #Rotate x-labels
    lineplot.set_xticklabels(rotation=30)
    #create title
    plt.title("Topics over time in r/WallStreetsBets")
    #set y label
    plt.ylabel("Topic dominance")
    #Set layout to be tight
    plt.tight_layout()
    #SHow plt figure (otherwise plt parameters won't be displayed)
    plt.show()
    #save figure
    plt.savefig("../output/Topic_over_time-Lineplot.png")
    
    """
    ----------- Make dataframe for csv output ----------
    """
    #Create dataframe with keywor
    df_topic_keywords = lda_utils.format_topics_sentences(ldamodel=lda_model, 
                                                          corpus=corpus, 
                                                          texts=data_processed)

    # Format
    df_dominant_topic = df_topic_keywords.reset_index()
    df_dominant_topic.columns = ['Chunk_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    
    # Display setting to show more characters in column
    pd.options.display.max_colwidth = 100

    sent_topics_sorteddf = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf = pd.concat([sent_topics_sorteddf, 
                                          grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                          axis=0)

    # Reset Index    
    sent_topics_sorteddf.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
    #save as csv
    sent_topics_sorteddf.to_csv("../output/topics_contribution.csv")
    
#Define behaviour when called from commandline
if __name__ == "__main__":
    main()
    