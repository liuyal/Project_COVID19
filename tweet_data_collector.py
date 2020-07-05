# ----------------------------------------------------------------------
# DATE: 2020/08/10
# AUTHOR: Jerry Liu
# EMAIL: Liuyal@sfu.ca
#
# DESCRIPTION:
# Tweet hydrate script to get daily 1000 random tweets
# ----------------------------------------------------------------------

import os
import sys
import time
import datetime
import shutil
import stat
import threading
import random
import tweepy
import copy
import requests
import queue


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
    CONSUMER_KEY = keys[0].split('"')[1]
    CONSUMER_SECRET = keys[1].split('"')[1]
    OAUTH_TOKEN = keys[2].split('"')[1]
    OAUTH_TOKEN_SECRET = keys[3].split('"')[1]
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    return tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


def curl_id(hydrate_directory, repo, start_date="2020-03-22", n=1):
    response = requests.get(repo)
    text_list = response.text.split('\n')

    folder_list = []
    for item in text_list:
        if "2020" in item and "js-navigation-open link-gray-dark" in item:
            folder_list.append("https://github.com" + item[item.index('href="') + len('href="'): item.index('"', item.index('href="') + len('href="'), -1)])

    file_list = {}
    for item in folder_list:
        response = requests.get(item)
        text_list = response.text.split('\n')
        for string in text_list:
            if "coronavirus-tweet-id" in string and "js-navigation-open link-gray-dark" in string:
                url = "https://raw.githubusercontent.com" + string[string.index('href="') + len('href="'): string.index('"', string.index('href="') + len('href="'), -1)]
                date = "-".join(string.split("tweet-id-")[-1].split(".txt")[0].split("-")[0:-1])
                if date not in list(file_list.keys()): file_list[date] = []
                file_list[date].append(url.replace("blob/", ''))

    if os.path.exists(os.getcwd() + os.sep + "data"):
        if os.path.exists(hydrate_directory):
            dates = [dates.replace(".csv", '') for dates in os.listdir(hydrate_directory)]
            repo_dates = [dates for dates in list(file_list.keys()) if dates > start_date]
            if len(repo_dates) != len(dates):
                delta = set(repo_dates).difference(set(dates))
                if len(delta) > 0:
                    filtered_file_list = {}
                    for item in list(delta):
                        filtered_file_list[item] = file_list[item]
                    file_list = filtered_file_list
                else:
                    file_list = {}
            else:
                file_list = {}

    q = queue.Queue()
    thread_list = []
    for item in file_list:
        if item > start_date:
            thread_list.append(threading.Thread(target=id_request, args=(item, file_list[item], q, n)))
    [item.start() for item in thread_list]
    [item.join() for item in thread_list]

    return list(q.queue)


def id_request(id, url_list, q, n=1):
    id_list = []
    for item in random.sample(url_list, n):
        response = requests.get(item)
        id_list = id_list + response.text.split('\n')
        sys.stdout.write(id + " " + item + " " + "\n")
    sys.stdout.write(id + " Complete!\n")
    q.put((id, id_list))


def hydrate(id_log, hydrate_directory, api, start_date="2020-03-22", n=10):
    if not os.path.exists(os.getcwd() + os.sep + "data"): os.makedirs(os.getcwd() + os.sep + "data")
    if not os.path.exists(hydrate_directory): os.mkdir(hydrate_directory)

    for date, id_list in id_log:
        counter = 0
        copy_list = copy.deepcopy(id_list)
        data_header = "index,id,create_at,text,user_name,verified,location,followers_count,extended,retweeted,quoted"
        file = open(hydrate_directory + os.sep + date + ".csv", "a+", encoding="utf8")
        file.truncate(0)
        file.write(data_header + "\n")

        while counter < n and date > start_date:
            data = []
            random.shuffle(copy_list)
            id = copy_list.pop(0)
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
                counter += 1
                file.write(",".join(data) + "\n")
                file.flush()
            except Exception as e:
                print("ERROR:", e)
        file.close()
        print(date, " hydrate Complete!")


if __name__ == "__main__":
    tweet_id_repo = r"https://github.com/echen102/COVID-19-TweetIDs"
    hydrate_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_hydrated_tweets"

    print("Curl COVID-19 GIT Tweet ID...")
    id_list = curl_id(hydrate_directory, tweet_id_repo, "2020-03-22", 1)

    print("Loading Twitter API KEYs...")
    api = get_token("jerry.token")

    print("Hydrating COVID-19 Tweet text from ID...")
    hydrate(id_list, hydrate_directory, api, "2020-03-22", 1000)