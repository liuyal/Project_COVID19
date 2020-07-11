# ----------------------------------------------------------------------
# DATE: 2020/08/10
# AUTHOR: Jerry Liu
# EMAIL: Liuyal@sfu.ca
#
# DESCRIPTION:
# COVID-19 Data Mining Project
# ----------------------------------------------------------------------

import os
import sys
import time
import datetime
import shutil
import threading
import stat
import copy
import queue
import re
import csv
import sqlite3
import itertools
import collections
import langdetect
import numpy as np
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import spacy
from spacy_langdetect import LanguageDetector
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer, SnowballStemmer


def print_header(path):
    f = open(path, "r", encoding="utf8")
    print(f.read())
    f.close()
    print("\nDATE: 2020/08/10")
    print("AUTHOR: Jerry Liu")
    print("EMAIL: Liuyal@sfu.ca")


def delete_folder(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.chmod(os.path.join(root, dir), stat.S_IRWXU)
        for file in files:
            os.chmod(os.path.join(root, file), stat.S_IRWXU)
    shutil.rmtree(path, ignore_errors=True)


def load_csv_data(file_directory):
    data_output = {}
    for item in os.listdir(file_directory):
        file = open(file_directory + os.sep + item, "r+", encoding="utf-8")
        data = csv.reader(file)
        for row in data:
            for i in range(0, len(row)):
                try:
                    row[i] = float(row[i])
                except:
                    row[i] = row[i]
            if item.split('.')[0] not in list(data_output.keys()):
                data_output[item.split('.')[0]] = []
            data_output[item.split('.')[0]].append(row)
        file.close()
    return data_output


def to_data_frame(data):
    data_frame_list = []
    for date in list(data.keys()):
        daily_data = data[date]
        header = daily_data.pop(0)
        df = pd.DataFrame(daily_data, columns=header)
        df.insert(0, "Date", [date] * len(daily_data), True)
        data_frame_list.append(df)
    data_frame = pd.concat(data_frame_list, axis=0, join='outer', sort=False, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
    return data_frame


def process_location_data(data_frame):
    daily_us = data_frame.loc[data_frame["Country_Region"] == "US"]
    aggregate = {"Confirmed": "sum", "Deaths": "sum", "Recovered": "sum"}
    daily_us_total = daily_us.groupby("Date").agg(aggregate).reset_index()
    return daily_us_total


def language_process(raw_text, nlp, words):
    # Functions are credited to TA Arjun Mahadevan
    url_pattern = re.compile(r'https://\S+|www\.\S+')
    replace_url = url_pattern.sub(r'', str(raw_text))
    # Remove url and punctuation base on regex pattern
    punctuation_pattern = re.compile(r'[^\w\s\-]')
    no_punctuation = punctuation_pattern.sub(r'', replace_url)
    processed_text = re.sub(r'^[0-9]*$', '', no_punctuation)
    # Load NLTK's words library and filter out non-english words
    processed_text = " ".join(w for w in nltk.wordpunct_tokenize(processed_text) if w.lower() in words)
    doc = nlp(processed_text.lower())
    # Tokenize text and remove words that are less than 3 letters and stop words
    output_words = [token.text for token in doc if token.is_stop is not True and token.is_punct is not True]
    output_words = [letters for letters in output_words if len(letters) > 2]
    word_collection = collections.Counter(output_words)
    return word_collection


def tokenize_mt_wrapper(data, nlp, words):
    for line in data:
        tokens = language_process(line[3], nlp, words)
        print(tokens)


def process_tweet_data(data_frame, nlp, words):
    # thread_list = []
    # for date in data_frame:
    #     tokenize_mt_wrapper(data_frame[date], nlp, words)
    tokenize_mt_wrapper(data_frame["2020-04-01"], nlp, words)


if __name__ == "__main__":
    # print_header(os.getcwd() + os.sep + "header.txt")

    location_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_location_data"
    tweet_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_filtered_tweets"

    np.random.seed(2020)
    nlp = spacy.load("en")
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
    nltk.download('wordnet')
    words = set(nltk.corpus.words.words())

    print("Loading COVID-19 datasets...")
    # location_data = load_csv_data(location_directory)
    tweet_data = load_csv_data(tweet_directory)

    # location_data_frame = to_data_frame(location_data)
    # tweet_data_frame = to_data_frame(tweet_data)

    # print("Processing Location Data...")
    # process_location_data(location_data_frame)

    print("Processing Tweet Data...")
    process_tweet_data(tweet_data, nlp, words)

    print()
