# ----------------------------------------------------------------------
# DATE: 2020/08/10
# AUTHOR: Jerry Liu
# EMAIL: Liuyal@sfu.ca
#
# DESCRIPTION:
# Tweet ID collector & hydration script
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
    if not os.path.exists(path): path = "twitter.token"
    file = open(path, "r")
    keys = file.readlines()
    file.close()
    CONSUMER_KEY = keys[0].split('"')[1]
    CONSUMER_SECRET = keys[1].split('"')[1]
    OAUTH_TOKEN = keys[2].split('"')[1]
    OAUTH_TOKEN_SECRET = keys[3].split('"')[1]
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    return tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


def id_request(date, url_list, q, n=1, verbose=False):
    id_list = []
    for item in random.sample(url_list, n):
        response = requests.get(item)
        id_list = id_list + response.text.split('\n')
        if verbose: sys.stdout.write(date + " " + item + " " + "\n")
    if verbose: sys.stdout.write(date + " Complete!\n")
    q.put((date, id_list))


def curl_id(repo, start_date, n, hydrate_directory):
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
    for date in file_list:
        if date > start_date:
            thread_list.append(threading.Thread(target=id_request, args=(date, file_list[date], q, n)))
    [item.start() for item in thread_list]
    [item.join() for item in thread_list]
    return list(q.queue)


def tweet_counter_mt_helper(date, data, q, verbose=False):
    id_count = 0
    for url in data:
        response = requests.get(url)
        id_count = id_count + len(response.text.split('\n'))
        if verbose: sys.stdout.write(date + " " + url + " " + str(id_count) + "\n")
    if verbose: sys.stdout.write(date + " Complete!\n")

    q.put((date, id_count))


def tweet_counter(repo, output_directory):
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

    if os.path.exists(output_directory):
        file = open(output_directory)
        text = file.readlines()
        text.pop(0)
        file.close()
        existing_date_list = []
        for line in text:
            existing_date_list.append(line.split(',')[0])
        delta = set(list(file_list.keys())).difference(set(existing_date_list))
    else:
        delta = []
        file = open(output_directory, "a+")
        file.truncate(0)
        file.write("date,tweet_count\n")
        file.flush()
        file.close()

    q = queue.Queue()
    thread_list = []
    for date in file_list:
        if date in delta:
            thread_list.append(threading.Thread(target=tweet_counter_mt_helper, args=(date, file_list[date], q)))
    [item.start() for item in thread_list]
    [item.join() for item in thread_list]
    results = list(q.queue)
    results.sort()

    file = open(output_directory, "a+")
    for date, tweet_count in results:
        file.write(date + ',' + str(tweet_count) + '\n')
    file.flush()
    file.close()


def hydrate(id_log, api, start_date, n, hydrate_directory, verbose=False):
    if not os.path.exists(os.getcwd() + os.sep + "data"): os.makedirs(os.getcwd() + os.sep + "data")
    if not os.path.exists(hydrate_directory): os.mkdir(hydrate_directory)

    for date, id_list in sorted(id_log):
        counter = 0
        copy_list = copy.deepcopy(id_list)
        data_header = "index,id,create_at,text,user_name,verified,location,followers_count,extended,retweeted,quoted"
        file = open(hydrate_directory + os.sep + date + ".csv", "a+", encoding="utf8")
        file.truncate(0)
        file.write(data_header + "\n")

        while counter <= n and date > start_date:
            data = []
            random.shuffle(copy_list)
            if len(copy_list) == 0: break
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
                data[3] = text.replace("\n", " ").replace(",", " ").replace("\r", "")
                counter += 1
                file.write(",".join(data) + "\n")
                file.flush()
            except Exception as e:
                if verbose: print("ERROR:", e)

        file.close()
        if verbose: print(date, " hydrate Complete!")


if __name__ == "__main__":
    tweet_id_repo = r"https://github.com/echen102/COVID-19-TweetIDs"
    hydrate_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_hydrated_tweets"
    tweet_count_result_directory = os.getcwd() + os.sep + "data" + os.sep + "tweet_count_result.csv"

    print("Checking COVID-19 Tweet ID GitHub REPO...")
    id_list = curl_id(tweet_id_repo, "2020-01-01", 1, hydrate_directory)

    print("Counting Number of Tweets...")
    tweet_counter(tweet_id_repo, tweet_count_result_directory)

    print("Loading Twitter API KEYs...")
    api = get_token("jerry.token")

    print("Hydrating COVID-19 Tweet Text...")
    hydrate(id_list, api, "2020-01-01", 1000, hydrate_directory)

    print("Tweet Collector Complete!")
