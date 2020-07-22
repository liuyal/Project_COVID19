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
import stat
import re
import copy
import queue
import threading
import itertools
import csv
import string
import random
import collections
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import spacy
import nltk
from nltk.stem.porter import *
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk import FreqDist
from nltk import classify


def print_header(path):
    f = open(path, "r", encoding="utf8")
    print(f.read())
    f.close()
    print("\nDATE: 2020/08/10")
    print("AUTHOR: Jerry Liu")
    print("EMAIL: Liuyal@sfu.ca\n")


def load_nltk_packages():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("words")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")


def dataset_check():
    # TODO: check repo
    os.system("python " + os.getcwd() + os.sep + "csse_data_collector.py")
    # TODO: check repo
    os.system("python " + os.getcwd() + os.sep + "tweet_data_collector.py")
    os.system("python " + os.getcwd() + os.sep + "tweet_data_filter.py")


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
    output_tokens = [letters for letters in output_words if len(letters) > 2]
    word_collection = collections.Counter(output_words)
    return output_tokens, word_collection


def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|''(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def tweet_model_generator(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def create_trainer_model(training_data_path, model_path, training_size=500000):
    if os.path.exists(model_path):
        loaded_model = pickle.load(open(model_path, 'rb'))
        return loaded_model

    file = open(training_data_path, "r", encoding='utf-8')
    raw_csv = csv.reader(file)
    raw_training_data = []
    for row in raw_csv: raw_training_data.append(row)
    file.close()

    raw_training_data.pop(0)
    random.shuffle(raw_training_data)
    if training_size > len(raw_training_data):
        raw_training_data = random.sample(raw_training_data, len(raw_training_data))
    else:
        raw_training_data = random.sample(raw_training_data, training_size)

    negative_cleaned_tokens_list = []
    neutral_cleaned_tokens_list = []
    positive_cleaned_tokens_list = []
    pbar = tqdm(total=len(raw_training_data))
    for line in raw_training_data:
        negative_score = line[3]
        neutral_score = line[4]
        positive_score = line[5]
        tokens = nltk.word_tokenize(line[2])
        if negative_score > positive_score and negative_score > neutral_score:
            negative_cleaned_tokens_list.append(remove_noise(tokens, stopwords.words('english')))
        elif neutral_score > positive_score and neutral_score > negative_score:
            neutral_cleaned_tokens_list.append(remove_noise(tokens, stopwords.words('english')))
        elif positive_score > neutral_score and positive_score > negative_score:
            positive_cleaned_tokens_list.append(remove_noise(tokens, stopwords.words('english')))
        pbar.update(1)
    pbar.close()

    negative_tokens_model = tweet_model_generator(negative_cleaned_tokens_list)
    neutral_tokens_model = tweet_model_generator(neutral_cleaned_tokens_list)
    positive_tokens_model = tweet_model_generator(positive_cleaned_tokens_list)
    negative_dataset = []
    neutral_dataset = []
    positive_dataset = []
    for tweet_dict in negative_tokens_model:
        negative_dataset.append((tweet_dict, "Negative"))
    for tweet_dict in neutral_tokens_model:
        neutral_dataset.append((tweet_dict, "Neutral"))
    for tweet_dict in positive_tokens_model:
        positive_dataset.append((tweet_dict, "Positive"))

    train_data = negative_dataset + neutral_dataset + positive_dataset
    classifier = NaiveBayesClassifier.train(train_data)
    if not os.path.exists(os.getcwd() + os.sep + "data" + os.sep + "classifier"):
        os.mkdir(os.getcwd() + os.sep + "data" + os.sep + "classifier")
    f = open(model_path, 'wb')
    pickle.dump(classifier, f)
    f.close()
    return classifier


def validate_classifier(classifier):
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stopwords.words('english')))
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stopwords.words('english')))

    result_list = []
    for tokens in positive_cleaned_tokens_list:
        result_list.append(classifier.classify(dict([token, True] for token in tokens)))
    print("Positive Tweet Detection Accuracy: " + str(100.0 * result_list.count("Positive") / len(result_list)) + "%")
    result_list = []
    for tokens in negative_cleaned_tokens_list:
        result_list.append(classifier.classify(dict([token, True] for token in tokens)))
    print("Negative Tweet Detection Accuracy: " + str(100.0 * result_list.count("Negative") / len(result_list)) + "%")


def sentiment_analyzer_mt_wrapper(date, data, classifier):
    result_list = []
    for line in data:
        if line[3] != "text":
            custom_tweet = line[3]
            custom_tokens = word_tokenize(custom_tweet)
            tokens = remove_noise(custom_tokens, stopwords.words('english'))
            result_list.append(classifier.classify(dict([token, True] for token in tokens)))
    return result_list


def tweet_sentiment_analyzer(data_frame, classifier, file_path, verbose=False):
    file = open(file_path, "a+")
    file.truncate(0)
    file.write("date,Negative,Neutral,Positive\n")
    file.close()
    for date in data_frame:
        result_list = sentiment_analyzer_mt_wrapper(date, data_frame[date], classifier)
        file = open(file_path, "a+")
        file.write(date + "," + str(result_list.count("Negative")) + "," + str(result_list.count("Neutral")) + "," + str(result_list.count("Positive")) + "\n")
        file.flush()
        file.close()
        if verbose:
            print(date, result_list.count("Negative"), result_list.count("Neutral"), result_list.count("Positive"))


def tweet_topic_modeling(data, num_topic, nlp, words, output_path):
    for date in data:
        text_data = []
        for line in data[date]:
            if line[3] != "text":
                tokens, token_count = language_process(line[3], nlp, words)
                text_data.append(tokens)
        dictionary = corpora.Dictionary(text_data)
        corpus = [dictionary.doc2bow(text) for text in text_data]
        lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topic, id2word=dictionary, passes=15)
        topics = lda_model.print_topics(num_words=5)
        print(date, topics)



if __name__ == "__main__":
    # print_header(os.getcwd() + os.sep + "header.txt")

    # print("Loading NLTK Data Packages...")
    # load_nltk_packages()

    # print("Checking for dataset updates...")
    # dataset_check()

    location_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_location_data"
    tweet_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_filtered_tweets"

    nlp = spacy.load("en")
    words = set(nltk.corpus.words.words())

    print("Loading COVID-19 related datasets...", end='')
    location_data = load_csv_data(location_directory)
    tweet_data = load_csv_data(tweet_directory)
    print("[Complete]")

    print("Creating Data Frames...", end='')
    # location_data_frame = to_data_frame(location_data)
    # tweet_data_frame = to_data_frame(tweet_data)
    print("[Complete]")

    print("Training/Loading Tweet Classifier Model...")
    training_data_path = os.getcwd() + os.sep + "data" + os.sep + "training_data_t4sa" + os.sep + "training_data_t4sa.csv"
    model_path = os.getcwd() + os.sep + "data" + os.sep + "classifier" + os.sep + "NaiveBayesClassifier_1M.pickle"
    # classifier = create_trainer_model(training_data_path, model_path, 9999999)

    print("Testing Classifier Model on Example Tweets...")
    # validate_classifier(classifier)

    print("Processing Sentiment Analyzer...", end='\n')
    # tweet_sentiment_analyzer(tweet_data, classifier, os.getcwd() + os.sep + "data" + os.sep + "tweet_sentiment_result.csv")

    print("Processing Topic Generation...", end='\n')
    tweet_topic_modeling(tweet_data, 1, nlp, words, os.getcwd() + os.sep + "data" + os.sep + "dictionary" + os.sep + "tweet_topic_model.csv")
