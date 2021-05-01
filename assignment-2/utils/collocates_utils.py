#!/usr/bin/env python

# import libraries
import re #regex
import string #string


#Tokenizer
def tokenize(input_string):

    input_string = input_string.lower() #make input lower case
    token_list = re.findall(r'\b\w+\b', input_string) #find all word characters inside a word boundary
    
    return token_list

# Concordance lines
# NOTE the function can either take a tokenized corpus or a non-tokenized copus with the parameter "tokenize"
# If it is True it will tokenize for you else it will expect that the text is already tokenized.
# This is done for later use so I can create custom tokenization if I wish to.
def kwic(text, keyword, window_size, tokenize = True):
    i = 0
    lines = []
    #If the text needs to be tokenized tokenize it
    if tokenize == True:
        text = tokenize(text)
    
    #For every token in the text
    for token in text:
        if token == keyword: #If the token is the keyword
            # Get window
            window_start = max(0, i - window_size) #If the index i negative make it zero (avoids indexing errors) 
            window_end = i + (window_size + 1)
            #Plus one because of zero indexing (otherwise I get one less word on the right side of target word)
            
            # This tests If window index is larger than the text index
            # This might happen if the target keyword is one of the last words in the corpus)
            # This is supposed to make sure, that I don't get an indexing error. 
            if window_end > (len(text) - 1):
                window_end = (len(text) - 1) #Make it only include up til the last word in the corpus
                    
            #Add line to output variable
            line = text[window_start:window_end]
            lines.append(line)   
        i+=1
    return lines

if __name__ == "__main__":
    pass #dont run from terminal