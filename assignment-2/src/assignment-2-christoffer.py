#!/usr/bin/env python
"""
---------------------------------- Import libraries-------------------------------------------
"""
import re #regex
import string #string
import math
import os
from pathlib import Path
import csv 
"""
---------------------------------- Parameters------------------------------------------
"""
keyword = "bald"
window_size= 55 
corpus_dir = os.path.join("..", "100_english_novels", "corpus") #path to directory
output = os.path.join("..", "output", "output.csv") #destination and name for output-file
"""
----------------------------------Functions-------------------------------------------
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
----------------------------------Main function-------------------------------------------
"""
def main():
    
    """
    ----------------------------------Create corpus-------------------------------------------
    """
    corpus = ""
    
    for file_name in Path(corpus_dir).glob("*.txt"):
        with open(file_name, "r", encoding="utf-8") as file: #open the file
            corpus = corpus + file.read()
    """
    ----------------------------------Tokenize corpus-------------------------------------------
    """
    tokenized_corpus = tokenize(corpus)
    tokenized_lines = kwic(corpus, keyword, window_size)
    """
    ----------------------------------Count collocates-------------------------------------------
    """
    #Create a list of collocates
    collocates = []
    index = 0
    for line in tokenized_lines:
        for token in tokenized_lines[index]:
            if token not in collocates:
                collocates.append(token)
        index = index + 1
        
    #Remove partial words
    for collocate in collocates:
        if collocate not in tokenized_corpus:
            collocates.remove(collocate)
            
    #Count collocates
    collocate_counts = []
    for collocate in collocates:
        count = 0
        for line in tokenized_lines:
            count = count + line.count(collocate)
        collocate_counts.append(count)
    """
    ----------------------------------Create csv and Calculate MI-------------------------------------------
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
            if E11 <= 0:
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