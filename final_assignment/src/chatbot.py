# data tools
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path

# tensorflow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# BERT
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

def main():

    
    #Create an argument parser from argparse
    ap = argparse.ArgumentParser(description = "[INFO] CNN for classifying houses in Game of Thrones") 
    
    # question
    ap.add_argument("-Q", "--question",
                    required = True,
                    type = str,
                    help = "[INFO] Type a question")
    #end it?
    ap.add_argument("-e", "--end",
                    required = False,
                    default = False,
                    type = bool,
                    help = "type --end True, to end the chatbot")
    #return arguments in parser
    args = vars(ap.parse_args())
    question = args["question"]
    end = args["end"]
    #print(question)
    with open(Path("../data/Das_Kapital_Volume_One.txt"), "r", encoding="utf-8") as file: #open the file
            paragraph = file.read()
            paragraph = str(paragraph)
            
    #import os
    #os.system('wget https://github.com/see--/natural-question-answering/releases/download/v0.0.1/tokenizer_tf2_qa.zip')
    #os.system('unzip tokenizer_tf2_qa.zip')
    
    tokenizer = BertTokenizer.from_pretrained('../data/tokenizer_tf2_qa/vocab.txt') #Download voacb
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1") # Download model with questions 
    
    question_tokens = tokenizer.tokenize(question)
    paragraph_tokens = tokenizer.tokenize(paragraph)
    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + paragraph_tokens + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(paragraph_tokens) + 1)
    
    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
    tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])
    # using `[1:]` will enforce an answer. `outputs[0][0][0]` is the ignored '[CLS]' token logit
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    print(f'Question: {question}')
    print(f'Answer: {answer}')
    print("\n")
if __name__ == "__main__":
    main()