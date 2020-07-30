# ----------------------------------------------------------------------
# DATE: 2020/08/10
# AUTHOR: Jerry Liu
# EMAIL: Liuyal@sfu.ca
# GITHUB: https://github.com/liuyal/Project_COVID19
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
import csv
import string
import random
import queue
import threading
import itertools
import collections
import pickle
import gensim
import spacy
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier
from nltk.stem import WordNetLemmatizer


def print_header(path):
    file = open(path, "r", encoding="utf8")
    print(file.read())
    file.close()
    print("\nDATE: 2020/08/10")
    print("AUTHOR: Jerry Liu")
    print("EMAIL: Liuyal@sfu.ca")
    print("LINK: https://github.com/liuyal/Project_COVID19\n")


def load_nltk_packages():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("words")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")


def dataset_check():
    os.system("python " + os.getcwd() + os.sep + "1_csse_data_collector.py")
    os.system("python " + os.getcwd() + os.sep + "2_tweet_data_collector.py")
    os.system("python " + os.getcwd() + os.sep + "3_tweet_data_filter.py")
    os.system("python " + os.getcwd() + os.sep + "4_tweet_data_tokenizer.py")


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
    file = open(model_path, 'wb')
    pickle.dump(classifier, file)
    file.close()
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
    token_list_daily = {}
    token_list_total = []
    for file_name in os.listdir(input_path):
        if "count" not in file_name.lower():
            date = file_name.split('_')[0]
            token_list_daily[date] = []
            file = open(input_path + os.sep + file_name)
            raw_text_lines = file.readlines()
            file.close()
            for line in raw_text_lines:
                token_list_daily[date].append(line.replace('\n', '').split(',')[1:])
                token_list_total.append(line.replace('\n', '').split(',')[1:])
    return token_list_daily, token_list_total


def tweet_daily_topic_modeling(num_topic, token_list, topic_output_path, dominant_output_path, verbose=False):
    if os.path.exists(topic_output_path) and os.path.exists(dominant_output_path):
        file = open(topic_output_path, "r+")
        output_dates_topic = file.readlines()
        file.close()
        file = open(dominant_output_path, "r+")
        output_dates_dominant = file.readlines()
        file.close()

        output_date_topic_list = []
        for line in output_dates_topic[1:]:
            output_date_topic_list.append(line.split(',')[0])

        output_date_dominant_list = []
        for line in output_dates_dominant[1:]:
            output_date_dominant_list.append(line.split(',')[0])

        input_date_list = []
        for date in list(token_list.keys()):
            input_date_list.append(date)

        for date in list(token_list.keys()):
            if date not in set(input_date_list).difference(set(output_date_topic_list + output_date_dominant_list)):
                del token_list[date]

    if not os.path.exists(topic_output_path):
        file = open(topic_output_path, "w+")
        file.truncate(0)
        file.write("date,index,topic_model\n")
        file.close()

    if not os.path.exists(dominant_output_path):
        file = open(dominant_output_path, "a+")
        file.truncate(0)
        file.write("date,perplexity,coherence_score,topic_index,percent_contribution,topic_keywords\n")
        file.close()

    for date in list(token_list.keys()):
        dictionary = corpora.Dictionary(token_list[date])
        corpus = [dictionary.doc2bow(text) for text in token_list[date]]
        lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topic, id2word=dictionary, passes=15, random_state=100, update_every=1, chunksize=100, alpha='auto', per_word_topics=True)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=token_list[date], dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()

        topics = lda_model.print_topics(num_words=num_topic)
        file = open(topic_output_path, "a+")
        for index, topic in topics:
            file.write(date + "," + str(index) + "," + str(topic) + "\n")
            if verbose:  print(date + " " + str(index) + " " + str(topic))
        file.flush()
        file.close()

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
        sent_topics_df.columns = ['topic_index', 'percent_contribution', 'topic_keywords']
        file = open(dominant_output_path, "a+")
        data = list(sent_topics_df.values.tolist())
        data.sort(key=lambda x: x[1], reverse=True)
        data = data[0]
        data[-1] = data[-1].replace(',', ';').replace(' ','')
        file.write(date + ',' + str(lda_model.log_perplexity(corpus)) + ',' + str(coherence_lda) + ',' + ','.join([str(x) for x in data]) + '\n')
        file.flush()
        file.close()
        if verbose: print(date, lda_model.log_perplexity(corpus), coherence_lda, data)


if __name__ == "__main__":
    # print_header(os.getcwd() + os.sep + "header.txt")
    #
    # print("Loading NLTK Data Packages...")
    # load_nltk_packages()
    #
    # print("Checking for dataset updates...")
    # dataset_check()

    nlp = spacy.load("en")
    words = set(nltk.corpus.words.words())

    tweet_filtered_data_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_filtered_tweets"
    tweet_sentiment_result_directory = os.getcwd() + os.sep + "data" + os.sep + "tweet_sentiment_result.csv"
    tweet_tokenized_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_tokenized_tweets"

    tweet_topic_modeling_result_daily_directory = os.getcwd() + os.sep + "data" + os.sep + "tweet_topic_modeling_result.csv"
    tweet_dominant_topic_result_daily_directory = os.getcwd() + os.sep + "data" + os.sep + "tweet_dominant_topic_result.csv"

    tweet_training_data_path = os.getcwd() + os.sep + "data" + os.sep + "training_data_t4sa" + os.sep + "training_data_t4sa.csv"
    tweet_classifier_model_path = os.getcwd() + os.sep + "data" + os.sep + "classifier" + os.sep + "NaiveBayesClassifier_1M.pickle"

    # print("Loading COVID-19 related datasets...")
    # tweet_data = load_csv_data(tweet_filtered_data_directory)
    #
    # print("Creating Data Frames...")
    # tweet_data_frame = to_data_frame(tweet_data)
    #
    # print("Training/Loading Tweet Classifier Model...")
    # classifier = create_classifier_model(tweet_training_data_path, tweet_classifier_model_path, 9999999)
    #
    # print("Testing Classifier Model on Example Tweets...")
    # validate_classifier(classifier)
    #
    # print("Processing Sentiment Analyzer...")
    # tweet_sentiment_analyzer(tweet_data, classifier, tweet_sentiment_result_directory, verbose=False)

    print("Processing Topic Modeling...")
    tweet_tokens_daily, tweet_tokens_total = load_tweet_tokens(tweet_tokenized_directory)
    tweet_daily_topic_modeling(10, tweet_tokens_daily, tweet_topic_modeling_result_daily_directory, tweet_dominant_topic_result_daily_directory, verbose=True)

    # os.system("python " + os.getcwd() + os.sep + "5_data_visualization.py")
