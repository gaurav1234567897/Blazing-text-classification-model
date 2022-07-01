#!/usr/bin/env python
# coding: utf-8

# function to preprocess
import re
import string
import pandas as pd
import numpy as np
#library that contains punctuation
import multiprocessing
from multiprocessing import Pool
import csv
import nltk
nltk.download('omw-1.4')
from nltk.stem import PorterStemmer
# nltk.download('wordnet')
ps = PorterStemmer()
nltk.download("punkt")
# Stop words removal
# importing nlp library
nltk.download('stopwords')
# Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')

def remove_punctuation(s):
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
    return s.translate(translator)

def stemmer(text):
    stem_text = " ".join([ps.stem(word) for word in nltk.word_tokenize(text)])
    return stem_text

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= " ".join([i for i in nltk.word_tokenize(text) if i not in stopwords])
    return output
        
def transform_instance(row):
    cur_row = []
    label = "__label__" + row[1]  # Prefix the index-ed label with __label__
    cur_row.append(label)
    cur_row.extend(nltk.word_tokenize(row[0].lower()))
#     cur_row.extend(nltk.word_tokenize(row[1].lower()))
    return cur_row

def preprocess(input_file, output_file):
    all_rows = []
    with open(input_file, "r") as csvinfile:
        csv_reader = csv.reader(csvinfile, delimiter=",")
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            row[0] = remove_stopwords(row[0])
            row[0] = remove_punctuation(row[0])
            row[0] = stemmer(row[0])
            all_rows.append(row)
    pool = Pool(processes=multiprocessing.cpu_count())
    transformed_rows = pool.map(transform_instance, all_rows)
    pool.close()
    pool.join()

    with open(output_file, "w") as csvoutfile:
        csv_writer = csv.writer(csvoutfile, delimiter=" ", lineterminator="\n")
        csv_writer.writerows(transformed_rows)

def preprocess_line(statements):
    all_rows = []
    for row in statements:
        row = remove_stopwords(row)
        row = remove_punctuation(row)
        row = stemmer(row)
        row = " ".join(nltk.word_tokenize(row.lower()))
        all_rows.append(row)
    return all_rows