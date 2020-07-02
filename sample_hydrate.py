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
import copy
import json
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
    f = open(path, "r")
    keys = f.readlines()
    f.close()

    return keys[0].split('"')[1], keys[1].split('"')[1], keys[2].split('"')[1], keys[3].split('"')[1]


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


def hydrate(local_data_directory, hydrate_directory, api, n=10):
    delete_folder(hydrate_directory)
    os.mkdir(hydrate_directory)

    id_log = {}
    for file in os.listdir(local_data_directory):
        f = open(local_data_directory + os.sep + file, "r")
        id_list = f.readlines()
        f.close()
        id_log[file.split('.')[0]] = id_list

    for date in id_log:
        counter = 0
        copy_list = copy.deepcopy(id_log[date])
        save_data = []
        while counter < n:

            data = []
            random.shuffle(copy_list)
            id = copy_list.pop(0)

            data_header = "index,id,create_at,text,user_name,verified,location,followers_count,extended,retweeted,quoted"
            file = open(hydrate_directory + os.sep + date + ".csv", "w+", encoding="utf8")
            file.truncate(0)
            file.write(data_header + "\n")
            file.flush()
            file.close()

            try:
                response = api.get_status(id)._json
                data.append(str(counter))
                data.append(str(response["id_str"]))
                data.append(str(response["created_at"]))
                data.append("")
                data.append(str(response["user"]["screen_name"]))
                data.append(str(response["user"]["verified"]))
                data.append(str(response["user"]["location"]).replace(",", "").replace("\n", " "))
                data.append(str(response["user"]["followers_count"]))

                text = str(response["text"])
                extended_tweet = 0
                retweeted_status = 0
                quoted_status = 0

                if "extended_tweet" in response.keys():
                    extended_tweet = 1
                    text = response["extended_tweet"]["full_text"]
                if "retweeted_status" in response.keys():
                    retweeted_status = 1
                    if "extended_tweet" in response["retweeted_status"].keys():
                        retweet = response["retweeted_status"]["extended_tweet"]["full_text"]
                    else:
                        retweet = response["retweeted_status"]["text"]
                    text = text + " " + retweet
                if "quoted_status" in response.keys():
                    quoted_status = 1
                    if "extended_tweet" in response["quoted_status"].keys():
                        quote = response["quoted_status"]["extended_tweet"]["full_text"]
                    else:
                        quote = response["quoted_status"]["text"]
                    text = text + " " + quote

                data.append(str(extended_tweet))
                data.append(str(retweeted_status))
                data.append(str(quoted_status))
                data[3] = text.replace("\n", " ").replace(",", " ")
                save_data.append(",".join(data))
                counter += 1
            except Exception as e:
                print("ERROR:", e)

        file = open(hydrate_directory + os.sep + date + ".csv", "a+", encoding="utf8")
        file.write("\n".join(save_data))
        file.flush()
        file.close()

        print(date)


if __name__ == "__main__":
    CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET = get_token(os.getcwd() + os.sep + "twitter.token")
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    remote_data_directory = r"\\192.168.1.100\share\data\COVID-19-TweetIDs"
    local_data_directory = os.getcwd() + os.sep + "data" + os.sep + "tweet_IDs"
    hydrate_directory = os.getcwd() + os.sep + "data" + os.sep + "hydrated_tweets"

    # collect_tweet_id(remote_data_directory, local_data_directory)

    hydrate(local_data_directory, hydrate_directory, api, 10)

    print("\nEOS")
