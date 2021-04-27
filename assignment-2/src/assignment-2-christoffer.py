#!/usr/bin/env python
"""
############### Import libraries ###############
"""
import re #regex
import string #string
import math
import os
from pathlib import Path
import csv
import argparse

"""
############### Functions ############### 
"""
#Tokenizer
def tokenize(input_string):

    input_string = input_string.lower()
    token_list = re.findall(r'\b\w+\b', input_string)
    
    return token_list

#Concordance lines
def kwic(text, keyword, window_size):
    lines = []
    for match in re.finditer(keyword, text):
        
        #First character of match
        word_start = match.start()
        #last character index of match
        word_end = match.end()
        
        #left window
        left_window_start = max(0, word_start - window_size)#If it is negative make it zero.
        left_window = text[left_window_start:word_start]
        
        #Right window
        right_window_end = word_end + window_size
        right_window = text[word_end : right_window_end]
        
        #Add line to output variable
        line = f"{left_window} {keyword} {right_window}"
        line = tokenize(line)
        lines.append(line)
       
        
    return lines
"""
############### Main function ###############
"""
def main():
    """
    ############### Parameters ###############
    """
    #Create argparser
    ap = argparse.ArgumentParser(description = "Finding collocates and calculating MI score in a corpus")
    # Keyword
    ap.add_argument("-kw", "--keyword",
                   required = False,
                   default = "bald",
                   type = str,
                   help = "[INFO] The keyword, you wish to find collocates for, [TYPE] int, [DEFAULT bald")
    #Window size
    ap.add_argument("-wz", "--window",
                    required = False,
                    default = 55,
                    type = int,
                    help = "[INFO] The size of the window in characters on each size. [TYPE] int, [DEFAULT] 55")
    #Path to corpus
    ap.add_argument("-c", "--corpus",
                    default = "../100_english_novels/corpus",
                    type = str,
                    help = """[INFO] the path to a corpus (needs to be a folder with txt-files) [TYPE] str, [DEFAULT]
                    ../100_english_novels/corpus""")
    
    #output
    ap.add_argument("-op", "--output",
                    default = "../output/output.csv",
                    type = str,
                    help = "[INFO] the path and filename to the output csv-file [TYPE] str, [DEFAULT] ../output/output.csv")
    
    #Return arguments
    args = vars(ap.parse_args())
    
    #Save in variables (for readability)
    keyword = args["keyword"]
    window_size= args["window"]
    corpus_dir = Path(args["corpus"]) #converting to path object so it works across different OS
    output = Path(args["output"])
    
    """
    ############### Create corpus ###############
    """
    corpus = ""
    
    for file_name in corpus_dir.glob("*.txt"):
        with open(file_name, "r", encoding="utf-8") as file: #open the file
            corpus = corpus + file.read() #Concatinate to one long string
    """
    ############### Tokenize corpus ###############
    """
    tokenized_corpus = tokenize(corpus) #Tokenize
    tokenized_lines = kwic(corpus, keyword, window_size) #Get KWIC lines
    """
    ############### Count collocates ###############
    """
    # Create a list of collocates
    collocates = []
    i = 0
    for line in tokenized_lines: # For every line in KWIC lines
        for token in tokenized_lines[i]: # For every token in the line
            if token not in collocates: # if the token isn't in the list of collocates
                collocates.append(token) # Append token to list of collocates
        i += 1
        
    # Remove partial words
    for collocate in collocates: # For every collocate in list of collocates
        if collocate not in tokenized_corpus: # If the collocate is not in the corpus
            collocates.remove(collocate) # Remove the collocate, since it is a partial word
            
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
        N = len(tokenized_corpus) #Length of corpus.
        u = tokenized_corpus.count(keyword) #how often does the keyword appear?   
        
        #Iterate over collates and calculate their MI value
        index = 0
        for collocate in collocates: 
            
            #Count how often the keyword appear with the collocate
            keyword_in_lines = 0
            for line in tokenized_lines:
                if collocate in line:
                    keyword_in_lines = line.count(keyword) + keyword_in_lines
              
            #calculate values
            v = tokenized_corpus.count(collocate) #How often does this collocate appear?
            O11 = collocate_counts[index] #How often does this collocate and keyword appear?
            O12 = u - keyword_in_lines #How often does the keyword appear without this collocate?
            O21 = v - O11 #How often does this collocate appear without keyword
            R1 = O11 + O12
            C1 = O11 + O21
            E11 = (R1*C1)/N 

            #If E11 is zero or below something is wrong, and the word should be ignored.
            if E11 == 0:
                continue
            
            else: #Else calculate MI
                MI = math.log(O11/E11)
                
                #write row in csv
                writer.writerow({"collocate": collocate, "raw_frequency": O11, "MI": MI})
                print(f"Word: {collocate}, frequency: {O11}, MI: {MI}")
            
            index = index + 1    
# Define behaviour when called from command line
if __name__=="__main__":
    main()