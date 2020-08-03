import os
import sys
import time
import csv


def check_training_data_folder(path):
    print(path)


def combine_tweet_sentiment(text_directory, sentiment_directory, output_directory):
    if os.path.exists(output_directory): return
    raw_tweets = []
    file = open(text_directory, "r+", encoding='utf-8')
    data = csv.reader(file)
    for row in data: raw_tweets.append(row)
    file.close()

    text_sentiment = []
    file = open(sentiment_directory, "r+", encoding='utf-8')
    t4sa = file.readlines()
    file.close()
    for line in t4sa: text_sentiment.append(line.replace('\n', '').split("\t"))

    raw_tweets.pop(0)
    text_sentiment.pop(0)
    raw_tweets_dict = {}
    for item in raw_tweets: raw_tweets_dict[item[0]] = item[1:]

    result = []
    for item in text_sentiment:
        packet = []
        packet.append(item[0])
        packet.append(raw_tweets_dict[item[0]][0].replace(',', ''))
        packet.append(item[1])
        packet.append(item[2])
        packet.append(item[3])
        result.append(packet)

    file = open(output_directory, 'a+', encoding='utf-8')
    file.truncate(0)
    file.write("INDEX,TWID,TEXT,NEG,NEU,POS\n")
    for i in range(0, len(result)):
        file.write(str(i) + ',' + ','.join(result[i]) + '\n')
    file.flush()
    file.close()


if __name__ == "__main__":
    training_data_directory = os.getcwd() + os.sep + "data" + os.sep + "training_data_t4sa"
    tweet_text_directory = os.getcwd() + os.sep + "data" + os.sep + "training_data_t4sa" + os.sep + "t4sa_raw_tweets_text.csv"
    sentiment_directory = os.getcwd() + os.sep + "data" + os.sep + "training_data_t4sa" + os.sep + "t4sa_text_sentiment.tsv"
    combine_file_directory = os.getcwd() + os.sep + "data" + os.sep + "training_data_t4sa" + os.sep + "t4sa_training_data.csv"

    print("Checking T4SA Training Data...")
    check_training_data_folder(training_data_directory)

    print("Extracting T4SA Sentiment Training Data...")
    combine_tweet_sentiment(tweet_text_directory, sentiment_directory, combine_file_directory)

    print("Training Data Extraction Complete!")
