import os
import sys
import time
import datetime
import shutil
import stat
import threading
import random
import tweepy
import csv
import requests
import queue
import spacy
from spacy_langdetect import LanguageDetector


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


def language_filter_mt_helper(date, data_list, q):
    sys.stdout.write("Processing [" + date + "]...\n")
    result_list = []
    for line in data_list:
        doc = nlp(line[3])
        result = doc._.language
        if (result["language"] == 'en' and result["score"] > 0.5) or 'text' == line[3]:
            result_list.append(line)
    q.put((date, result_list))
    sys.stdout.write(date + " Complete!\n")


def tweet_language_filter(tweet_data):
    output = queue.Queue()
    thread_list = []
    for date in list(tweet_data.keys()):
        thread_list.append(threading.Thread(target=language_filter_mt_helper, args=(date, list(tweet_data[date]), output)))
    [item.start() for item in thread_list]
    [item.join() for item in thread_list]
    return output.queue


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    tweet_id_repo = r"https://github.com/echen102/COVID-19-TweetIDs"
    hydrate_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_hydrated_tweets"
    filtered_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_filtered_tweets"

    tweet_data = load_csv_data(hydrate_directory)
    tweet_data_filtered = tweet_language_filter(tweet_data)
