import os
import sys
import time
import datetime
import shutil
import stat
import queue
import threading
import csv
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


def language_filter_mt_helper(output_directory, date, data_list):
    sys.stdout.write("Processing [" + date + "]...\n")
    file = open(output_directory + os.sep + date + ".csv", "a+", encoding='utf-8')
    file.truncate(0)
    for line in data_list:
        try:
            doc = nlp(line[3])
            result = doc._.language
            if (result["language"] == 'en' and result["score"] > 0.5) or 'text' == line[3]:
                sys.stdout.write("[" + date + "] " + ",".join(str(i) for i in line) + "\n")
                file.write(",".join(str(i) for i in line) + "\n")
                file.flush()
        except Exception as e:
            sys.stdout.write("ERROR: " + str(e))
    file.close()
    sys.stdout.write(date + " Complete!\n")


def tweet_language_filter(output_directory, tweet_data):
    delete_folder(output_directory)
    os.mkdir(output_directory)
    thread_list = []
    for date in list(tweet_data.keys()):
        thread_list.append(threading.Thread(target=language_filter_mt_helper, args=(output_directory, date, list(tweet_data[date]))))
    [item.start() for item in thread_list]
    [item.join() for item in thread_list]


if __name__ == "__main__":
    nlp = spacy.load("en")
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    tweet_id_repo = r"https://github.com/echen102/COVID-19-TweetIDs"
    hydrate_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_hydrated_tweets"
    filtered_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_filtered_tweets"

    tweet_data = load_csv_data(hydrate_directory)
    tweet_language_filter(filtered_directory, tweet_data)
