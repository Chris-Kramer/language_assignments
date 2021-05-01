#!/usr/bin/env python
"""
############### Import libraries ###############
"""
#system tools 
import os
import sys
sys.path.append(os.path.join(".."))
from pathlib import Path

#standard tools
import math
import csv
import argparse
from argparse import RawTextHelpFormatter # Formatting -help

#Homebrewed functions
import utils.collocates_utils as cu

"""
############### Main function ###############
"""
def main():
    """
    ############### Parameters ###############
    """
    #Create argparser
    ap = argparse.ArgumentParser(description = "Finding collocates and calculating MI score in a corpus",
                                 formatter_class=RawTextHelpFormatter)
    # Keyword
    ap.add_argument("-kw", "--keyword",
                   required = False,
                   default = "bald",
                   type = str,
                   help =
                    "[INFO] The keyword you wish to find collocates for \n"
                    "[TYPE] str \n"
                    "[DEFAULT] bald \n"
                    "[EXAMPLE] --keyword single")
    #Window size
    ap.add_argument("-wz", "--window_size",
                    required = False,
                    default = 5,
                    type = int,
                    help =
                    "[INFO] The size of the window." 
                    "[INFO] The value represents amount of words on each side of target word \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 5 \n"
                    "[EXAMPLE] --window_size 10")
    #Path to corpus
    ap.add_argument("-c", "--corpus",
                    required = False,
                    default = "100_english_novels",
                    type = str,
                    help = 
                    "[INFO] the folder name of the corpus \n" 
                    "[INFO] It must be a folder with txt-files and it must be placed within the 'data' folder \n" 
                    "[TYPE] str \n"
                    "[DEFAULT] 100_english_novels \n"
                    "[EXAMPLE] --corpus corpus_name")
    
    #output
    ap.add_argument("-op", "--output",
                    required = False,
                    default = "output.csv",
                    type = str,
                    help = 
                    "[INFO] the filename for the output csv-file. It must end in .csv \n"
                    "[TYPE] str \n"
                    "[DEFAULT] output.csv \n"
                    "[EXAMPLE] --output single_collocates.csv")
    
    #Return arguments
    args = vars(ap.parse_args())
    
    #Save in variables (for readability)
    keyword = args["keyword"]
    window_size= args["window_size"]
    corpus_dir = os.path.join("..", "data", args["corpus"])
    output = os.path.join("..", "output", args["output"])
    
    """
    ############### Create corpus, tokenize and create KWIC lines ###############
    """
    # --- Create corpus ------
    corpus = ""
    for file_name in Path(corpus_dir).glob("*.txt"):
        with open(file_name, "r", encoding="utf-8") as file: #open the file
            corpus = corpus + file.read() #Concatinate to one long string
    
    # --- Tokenize and create kwic lines ---
    tokenized_corpus = cu.tokenize(corpus) #Tokenize corpus
    # Get concordance lines
    tokenized_lines = cu.kwic(tokenized_corpus, keyword, window_size, tokenize = False) 
    # The last parameter indicates that my corpus is already tokenized, so the function shouldn't tokenize for me. 
    """
    ############### Count collocates ###############
    """
    # Create a list of collocates
    collocates = []
    i = 0
    for line in tokenized_lines: # For every line in KWIC lines
        for token in tokenized_lines[i]: # For every token in the line
            # If the token is the keyword skip it
            if token == keyword:
                continue
            # Else if the token isn't in the list of collocates    
            elif token not in collocates:
                collocates.append(token) # Append token to list of collocates
        i += 1
            
    # Count collocates
    collocate_counts = [] 
    for collocate in collocates: # For every collocate
        count = 0 # Create count variable
        for line in tokenized_lines: # For every line in the KWIC lines
            count = count + line.count(collocate) # Count how often the collocate appears and save in variable 
        collocate_counts.append(count) # Apend the total count to collocate_counts
    """
    ############### Create csv and Calculate MI ###############
    """  
    #Create csv
    with open(output, mode = "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["collocate", "raw_frequency", "MI"])
        writer.writeheader()   
        
        #Static values
        N = len(tokenized_corpus) #Length of corpus
        u = tokenized_corpus.count(keyword) #how often does the keyword appear?   
        
        #Iterate over collates and calculate their MI value
        index = 0
        #for every collocate
        for collocate in collocates: 
            #Count how often the keyword appear with the collocate
            keyword_in_lines = 0
            for line in tokenized_lines: #for every line in KWIC lines
                if collocate in line: #If the collocate is in the line
                    keyword_in_lines = line.count(keyword) + keyword_in_lines #update count
              
            #calculate values
            v = tokenized_corpus.count(collocate) #How often does this collocate appear?
            O11 = collocate_counts[index] #How often does this collocate and keyword appear? (the keyword always appears with the collocate)
            O12 = u - keyword_in_lines #How often does the keyword appear without this collocate?
            O21 = v - O11 #How often does this collocate appear without keyword
            R1 = O11 + O12
            C1 = O11 + O21
            E11 = (R1*C1)/N 
            MI = math.log(O11/E11)
                
            #write row in csv
            writer.writerow({"collocate": collocate, "raw_frequency": O11, "MI": MI})
            print(f"Word: {collocate}, frequency: {O11}, MI: {MI}")
            
            index += 1   
# Define behaviour when called from command line
if __name__=="__main__":
    main()