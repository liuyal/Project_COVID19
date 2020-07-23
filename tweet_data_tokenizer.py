import os
import sys
import time
import datetime
import shutil
import stat
import re
import collections
import csv
import spacy
import nltk
from nltk.stem.porter import *


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


def tweet_tokenize(data, nlp, words, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        dates = []
        file_dates = []
        for date in data: dates.append(date.split(',')[0])
        for file in os.listdir(output_path): file_dates.append(file.split('_')[0])
        delta = set(dates).difference(set(file_dates))
        for item in list(data.keys()):
            if item not in delta:
                del data[item]

    token_count_total = []
    for date in data:
        token_list = []
        token_count = []
        file = open(output_path + os.sep + date + "_tokens.csv", "a+")
        file.truncate(0)
        file.flush()
        file.close()
        file = open(output_path + os.sep + date + "_tokens_count.csv", "a+")
        file.truncate(0)
        file.write("word,count\n")
        file.flush()
        file.close()

        for line in data[date]:
            if line[3] != "text":
                tokens, count = language_process(line[3], nlp, words)
                token_list.append((line[0], tokens))
                token_count.append(count)
                token_count_total.append(count)

        file = open(output_path + os.sep + date + "_tokens.csv", "a+")
        for index, tokens in token_list:
            if len(tokens) > 0:
                file.write(str(index) + "," + ",".join(tokens) + "\n")
        file.flush()
        file.close()

        file = open(output_path + os.sep + date + "_tokens_count.csv", "a+")
        collection_sum = sum(token_count, collections.Counter())
        for word, count in collection_sum.most_common():
            if len(word) > 2:
                file.write(word + "," + str(count) + "\n")
        file.flush()
        file.close()
        print(date, "Completed!")


if __name__ == "__main__":
    tweet_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_filtered_tweets"
    tweet_data = load_csv_data(tweet_directory)
    nlp = spacy.load("en")
    words = set(nltk.corpus.words.words())

    print("Tokenizing Filtered Tweets...", end='')
    tweet_tokenize(tweet_data, nlp, words, os.getcwd() + os.sep + "data" + os.sep + "covid_19_tokenized_tweets")
    print("[Complete!]")