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
import spacy
import nltk
import numpy as np
import pandas as pd
from spacy_langdetect import LanguageDetector


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
    return data_frame_list


def language_process(raw_text, nlp, words):
    # Functions are credited to TA Arjun Mahadevan
    url_pattern = re.compile(r'https://\S+|www\.\S+')
    replace_url = url_pattern.sub(r'', str(raw_text))
    # Remove url and punctuation base on regex pattern
    punctuation_pattern = re.compile(r'[^\w\s\-]')
    no_punctuation = punctuation_pattern.sub(r'', replace_url).lower()
    processed_text = re.sub(r'^[0-9]*$', '', no_punctuation)
    # Load NLTK's words library and filter out non-english words
    processed_text = " ".join(w for w in nltk.wordpunct_tokenize(processed_text) if w.lower() in words)
    doc = nlp(processed_text)
    # Tokenize text and remove words that are less than 3 letters
    output_words = [token.text for token in doc if token.is_stop is not True and token.is_punct is not True]
    output_words = [letters for letters in output_words if len(letters) > 2]
    word_collection = collections.Counter(output_words)

    return word_collection



if __name__ == "__main__":
    # print_header(os.getcwd() + os.sep + "header.txt")
    print("\nSTART Time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    location_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_location_data"
    tweet_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_hydrated_tweets"

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
    words = set(nltk.corpus.words.words())

    print("Loading COVID-19 datasets...")
    location_data = load_csv_data(location_directory)
    tweet_data = load_csv_data(tweet_directory)

    print("\nEND Time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

