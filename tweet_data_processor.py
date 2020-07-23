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
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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


def create_classifier_model(training_data_path, model_path, training_size=500000):
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

    if not os.path.exists(os.sep.join(model_path.split(os.sep)[0:-1])):
        os.mkdir(os.sep.join(model_path.split(os.sep)[0:-1]))
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


def tweet_sentiment_analyzer(data, classifier, file_path, verbose=False):
    file = open(file_path, "a+")
    file.truncate(0)
    file.write("date,Negative,Neutral,Positive\n")
    file.close()
    for date in data:
        result_list = []
        for line in data[date]:
            if line[3] != "text":
                custom_tweet = line[3]
                custom_tokens = word_tokenize(custom_tweet)
                tokens = remove_noise(custom_tokens, stopwords.words('english'))
                result_list.append(classifier.classify(dict([token, True] for token in tokens)))
        file = open(file_path, "a+")
        file.write(date + "," + str(result_list.count("Negative")) + "," + str(result_list.count("Neutral")) + "," + str(result_list.count("Positive")) + "\n")
        file.flush()
        file.close()
        if verbose:
            print(date, result_list.count("Negative"), result_list.count("Neutral"), result_list.count("Positive"))


def load_tweet_tokens(input_path):
    token_list = {}
    for file_name in os.listdir(input_path):
        if "count" not in file_name.lower():
            date = file_name.split('_')[0]
            token_list[date] = []
            f = open(input_path + os.sep + file_name)
            raw_text_lines = f.readlines()
            f.close()
            for line in raw_text_lines:
                token_list[date].append(line.replace('\n', '').split(',')[1:])
    return token_list


def tweet_topic_modeling(num_topic, token_list, output_path):
    if not os.path.exists(output_path):
        file = open(output_path, "w+")
        file.truncate(0)
        file.write("date,index,topic_model\n")
        file.flush()
        file.close()
    else:
        file = open(output_path, "r+")
        output_dates = file.readlines()
        file.close()

        output_date_list = []
        for line in output_dates[1:]:
            output_date_list.append(line.split(',')[0])
        input_date_list = []
        for date in list(token_list.keys()):
            input_date_list.append(date)

        for date in list(token_list.keys()):
            if date not in set(input_date_list).difference(set(output_date_list)):
                del token_list[date]

    for date in list(token_list.keys()):
        dictionary = corpora.Dictionary(token_list[date])
        corpus = [dictionary.doc2bow(text) for text in token_list[date]]
        lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topic, id2word=dictionary, passes=15)
        topics = lda_model.print_topics(num_words=5)
        file = open(output_path, "a+")
        for index, topic in topics:
            file.write(date + "," + str(index) + "," + str(topic) + "\n")
            print(date + " " + str(index) + " " + str(topic))
        file.flush()
        file.close()


def tweet_dominant_topic(num_topic, token_list, output_path):
    for date in token_list:

        dictionary = corpora.Dictionary(token_list[date])
        corpus = [dictionary.doc2bow(text) for text in token_list[date]]
        lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topic, id2word=dictionary, passes=15)

        sent_topics_df = pd.DataFrame()
        for i, row_list in enumerate(lda_model[corpus]):
            row = row_list[0] if lda_model.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:
                    wp = lda_model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        print()


def tweet_wordcount_frequency_distribution(input_path, output_path):
    text_list = []
    for file in os.listdir(input_path):
        if "count" in file:
            f = open(input_path + os.sep + file)
            text = f.readlines()
            text.pop(0)
            f.close()
            text_list = text_list + text


if __name__ == "__main__":
    # print_header(os.getcwd() + os.sep + "header.txt")
    # print("Loading NLTK Data Packages...")
    # load_nltk_packages()
    # print("Checking for dataset updates...")
    # dataset_check()

    # nlp = spacy.load("en")
    # words = set(nltk.corpus.words.words())

    location_data_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_location_data"
    tweet_filtered_data_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_filtered_tweets"
    tweet_sentiment_result_directory = os.getcwd() + os.sep + "data" + os.sep + "tweet_sentiment_result.csv"
    tweet_tokenized_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_tokenized_tweets"
    tweet_topic_modeling_result_directory = os.getcwd() + os.sep + "data" + os.sep + "tweet_topic_modeling_result.csv"
    tweet_dominant_topic_result_directory = os.getcwd() + os.sep + "data" + os.sep + "tweet_dominant_topic_result.csv"
    tweet_token_distribution_directory = os.getcwd() + os.sep + "data" + os.sep + "tweet_token_distribution.csv"
    training_data_path = os.getcwd() + os.sep + "data" + os.sep + "training_data_t4sa" + os.sep + "training_data_t4sa.csv"
    classifier_model_path = os.getcwd() + os.sep + "data" + os.sep + "classifier" + os.sep + "NaiveBayesClassifier_1M.pickle"

    # print("Loading COVID-19 related datasets...")
    # location_data = load_csv_data(location_data_directory)
    # tweet_data = load_csv_data(tweet_filtered_data_directory)
    #
    # print("Creating Data Frames...")
    # location_data_frame = to_data_frame(location_data)
    # tweet_data_frame = to_data_frame(tweet_data)
    #
    # print("Training/Loading Tweet Classifier Model...")
    # classifier = create_classifier_model(training_data_path, classifier_model_path, 9999999)
    #
    # print("Testing Classifier Model on Example Tweets...")
    # validate_classifier(classifier)
    #
    # print("Processing Sentiment Analyzer...")
    # tweet_sentiment_analyzer(tweet_data, classifier, tweet_sentiment_result_directory, verbose=False)

    print("Processing Topic Modeling...")
    tweet_token_list = load_tweet_tokens(tweet_tokenized_directory)
    # tweet_topic_modeling(5, tweet_token_list, tweet_topic_modeling_result_directory)
    tweet_dominant_topic(5, tweet_token_list, tweet_dominant_topic_result_directory)
    # tweet_wordcount_frequency_distribution(tweet_tokenized_directory, tweet_token_distribution_directory)
