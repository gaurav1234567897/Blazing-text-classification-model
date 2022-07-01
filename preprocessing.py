import subprocess
import logging
import pathlib
import requests
import tempfile
import boto3
import pandas as pd
# function to preprocess
import re
import string
#library that contains punctuation
import multiprocessing
from multiprocessing import Pool
import csv
import numpy as np
subprocess.run("pip install nltk",shell=True)
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download("punkt")
from nltk.stem import WordNetLemmatizer
# defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

# Stop words removal
# importing nlp library
nltk.download('stopwords')
# Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def remove_punctuation(s):
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
    return s.translate(translator)

def lemmatizer(text):
    lemm_text = " ".join([wordnet_lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)])
    return lemm_text

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= " ".join([i for i in nltk.word_tokenize(text) if i not in stopwords])
    return output
        
def transform_instance(row):
    cur_row = []
    label = "__label__" + row[1]  # Prefix the index-ed label with __label__
    cur_row.append(label)
    cur_row.extend(nltk.word_tokenize(row[0].lower()))
#     cur_row.extend(nltk.word_tokenize(row[2].lower()))
    return cur_row

def preprocessing(input_file, output_file):
    all_rows = []
    with open(input_file, "r") as csvinfile:
        csv_reader = csv.reader(csvinfile, delimiter=",")
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            row[0] = remove_stopwords(row[0])
            row[0] = remove_punctuation(row[0])
            row[0] = lemmatizer(row[0])
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
        row = lemmatizer(row)
        row = " ".join(nltk.word_tokenize(row.lower()))
        all_rows.append(row)
    return all_rows

if __name__ == "__main__":
    
    logger.debug("Starting preprocessing.")
    
    base_dir = "/opt/ml/processing"
    bucket = "sagemaker-us-west-2-430758128697"
    key = "ICM"

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data"
    pathlib.Path(fn).mkdir(parents=True, exist_ok=True)
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key+'/train.csv', f"{fn}/train.csv")
    s3.Bucket(bucket).download_file(key+'/validation.csv', f"{fn}/validation.csv")
    s3.Bucket(bucket).download_file(key+'/test.csv', f"{fn}/test.csv")
    
    logger.debug("Reading and processing all data.")
   
    # Preparing the training dataset
    preprocessing(f"{fn}/train.csv", f"{base_dir}/train/train.txt")
    
    # Preparing the validation dataset
    preprocessing(f"{fn}/validation.csv", f"{base_dir}/validation/validation.txt")
    
    # Preparing the testing dataset
    test = pd.read_csv(f"{fn}/test.csv")
    test["Text"] = preprocess_line(test["Text"])
    test.to_csv(f"{fn}/test.csv", index=False) 
