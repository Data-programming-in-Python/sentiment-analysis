# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""
import os
import pandas as pd

import json

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource, request
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split

import numpy as np
import re
from stop_words import stop_words
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

port = int(os.getenv('PORT', '5000'))

app = Flask(__name__)
api = Api(app)


# argument parsing
#parser = reqparse.RequestParser()
#parser.add_argument('query')

#text cleaning functions for running the text tester
def lower_case(line): return line.lower().strip()

def stem_words(line):
    ps = PorterStemmer()

    words = line.split()
    
    return_list = [ps.stem(word.strip()) for word in words]

    return ' '.join(return_list)

def remove_stop_words(line):

    words = line.split()
    
    kept_words = [word for word in words if word not in stop_words]

    return ' '.join(kept_words)

def remove_special_characters_and_numbers(line):
    return re.sub(r'([^a-zA-Z\s]+?)', '', line)

def get_irrelevant_words():
    
    # irrelevant words list was generated in a separate script that evaluated whether words were either
    # very low occurrence, or were similiarly represented in both positive and negative reviews.
    irrelevant_words_file = open('irrelevant_words.txt')    
    lines = irrelevant_words_file.readlines()
    irrelevant_words_set = {word.strip() for word in lines}
    irrelevant_words_file.close()
    
    return irrelevant_words_set


#prepare irrelevant words list
irrelevant_words = get_irrelevant_words()

def remove_irrelevant_words(line):
    words = line.split()
    kept_words = [word for word in words if word not in irrelevant_words]
    return ' '.join(kept_words)

def get_words_set(df): 
    df.dropna()
    word_set = set()
    for index, row in df.iterrows():
        try:
            review_words = row['Review'].split()
        except:
            continue
        for word in review_words:
            word = word.strip()
            if word not in irrelevant_words:
                word_set.add(word)
                
    return word_set

#prepare global vars
rootcleaned = pd.read_csv('cleaned.csv')
wordset = get_words_set(rootcleaned)
wordset.add('_Freshness') 

def create_row_dict(index, row, word_set):
    
    try:
        row_words = set(row['Review'].split())
    except:
        row_words = set()
    
    return_dict = {header: (0, 1)[header in row_words] for header in word_set}
    #return_dict['_Freshness'] = row['Freshness']
    return return_dict


def vectorize(df):
           
    dict_list = [create_row_dict(index, row, wordset) for index, row in df.iterrows()]

    return_df = pd.DataFrame(dict_list)

    #print(return_df.head())
    return return_df

def clean_data(df):
    df['Review'] = df['Review'].apply(lower_case)
    #print('Finished, lower_case: ')
    #get_time()
    df['Review'] = df['Review'].apply(remove_stop_words)
    #print('Finished, remove_stop_words: ')
    #get_time()
    df['Review'] = df['Review'].apply(remove_special_characters_and_numbers)
    #print('Finished, remove_special_characters_and_numbers: ')
    #get_time()
    df['Review'] = df['Review'].apply(stem_words)
    #print('Finished, stem_words: ')
    #get_time()
    
    df['Review'] = df['Review'].apply(remove_irrelevant_words)
    #print('Finished, remove_irrelevant_words: ')
    #get_time()
    
    df['Review'].replace('', np.nan, inplace=True)
    df.dropna(subset=['Review'], inplace=True)
    return df

def prepare_input_text(in_df):
    # This method will get the data in the correct format for testing the model
    cd = clean_data(in_df)
    vectorized = vectorize(cd)
    return vectorized

class SentimentAnalysis(Resource):
    def get(self):
        model_path = './clf_model.pkl'
        
        with open(model_path, 'rb') as f:
            ets_model = joblib.load(f)
                
        # get the query parameters
        intext = request.args.get('testtext')
        
        # build a data frame for sending the input to the model
        dfTest = pd.DataFrame()

        dfTest = dfTest.append({"Review":intext}, ignore_index=True)
        scrubbedInputTest = prepare_input_text(dfTest)
        
        # make a prediction
        pred_uc = ets_model.predict(scrubbedInputTest)
       
        # create JSON object
        output = str(pred_uc[0])
        
        return output
    

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(SentimentAnalysis, '/sentanalysis')



if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=port)