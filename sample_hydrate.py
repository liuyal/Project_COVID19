# ----------------------------------------------------------------------
# DATE: 2020/08/10
# AUTHOR: Jerry Liu
# EMAIL: Liuyal@sfu.ca
#
# DESCRIPTION:
# Tweet hydrator daily 1000 random
# ----------------------------------------------------------------------

import os
import sys
import time
import datetime
import shutil
import stat
import paramiko
import threading
import random
import tweepy
from tqdm import tqdm
from twarc import Twarc


def delete_folder(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.chmod(os.path.join(root, dir), stat.S_IRWXU)
        for file in files:
            os.chmod(os.path.join(root, file), stat.S_IRWXU)
    shutil.rmtree(path, ignore_errors=True)


def get_token(path):


    return 0, 0, 0, 0



def save_id(id, data_files):
    for file in data_files:
        date = '-'.join(file.split(os.sep)[-1].split('.')[0].split('id-')[-1].split('-')[0:-1])
        f1 = open(file, "r")
        id_list = f1.read()
        f1.close()
        f2 = open(os.getcwd() + os.sep + "data" + os.sep + "tweet_IDs" + os.sep + date + ".txt", "a+")
        f2.write(id_list)
        f2.flush()
        f2.close()


def collect_tweet_id(remote_directory, local_directory):
    data_files = {}
    for item in os.listdir(remote_directory):
        if os.path.isdir(remote_directory + os.sep + item) and "-" in item:
            for file in os.listdir(remote_directory + os.sep + item):
                date = '-'.join(file.split(os.sep)[-1].split('.')[0].split('id-')[-1].split('-')[0:-1])
                if date not in list(data_files.keys()): data_files[date] = []
                data_files[date].append(remote_directory + os.sep + item + os.sep + file)

    delete_folder(local_directory)
    os.mkdir(local_directory)

    thread_list = []
    for key in list(data_files.keys()):
        input_list = list(data_files[key])
        thread_list.append(threading.Thread(target=save_id, args=(key, input_list)))
    [item.start() for item in thread_list]
    [item.join() for item in thread_list]


def sample_id(path, n):
    id_log = {}
    for file in os.listdir(path):
        f = open(path + os.sep + file, "r")
        id_list = f.readlines()
        f.close()
        sample = random.sample(id_list, n)
        id_log[file.split('.')[0]] = sample

    return id_log


def hydrate(directory, ids, api):

    delete_folder(directory)
    os.mkdir(directory)

    for date in ids:

        for id in ids[date]:




            print(date, id)



if __name__ == "__main__":

    CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET = get_token(os.getcwd() + os.sep + "twitter.token")
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)


    remote_data_directory = r"\\192.168.1.100\share\data\COVID-19-TweetIDs"
    local_data_directory = os.getcwd() + os.sep + "data" + os.sep + "tweet_IDs"
    hydrate_directory = os.getcwd() + os.sep + "data" + os.sep + "hydrated_tweets"

    # collect_tweet_id(remote_data_directory, local_data_directory)

    sampled_ids = sample_id(os.getcwd() + os.sep + "data" + os.sep + "tweet_IDs", 50)

    hydrate(hydrate_directory, sampled_ids, api)

    print("\nEOS")
