#!/usr/bin/env python

# import libraries
import re #regex
import string #string


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

if __name__ == "__main__":
    pass #dont run from terminal