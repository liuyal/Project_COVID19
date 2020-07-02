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
import csv
import subprocess
import shutil
import stat
import numpy as np
import pandas as pd
import itertools
import sqlite3


def delete_folder(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.chmod(os.path.join(root, dir), stat.S_IRWXU)
        for file in files:
            os.chmod(os.path.join(root, file), stat.S_IRWXU)
    shutil.rmtree(path, ignore_errors=True)


def load_csv_data(file_name):
    data_output = []
    file = open(file_name, "r+", encoding="utf-8")
    data = csv.reader(file)
    for row in data:
        for i in range(0, len(row)):
            if row[i].isnumeric():
                row[i] = float(row[i])
        data_output.append(row)
    file.close()
    return data_output


def check_repo_data(repos):
    if not os.path.exists(os.getcwd() + os.sep + "data"):

        for folder in os.listdir(os.getcwd()):
            if os.path.isdir(folder) and ".idea" not in folder and ".git" not in folder and "data" not in folder:
                delete_folder(os.getcwd() + os.sep + folder)

        for repo, src_path in repos: os.system("git clone " + repo)
        os.makedirs(os.getcwd() + os.sep + "data")

        for repo, src_path in repos:
            dst = os.getcwd() + os.sep + "data" + os.sep + repo.split('/')[-1].split('.')[0].lower()
            if not os.path.isdir(dst): os.mkdir(dst)
            for root, dirs, files in os.walk(src_path, topdown=False):
                for name in files:
                    if ".csv" in name:
                        shutil.copyfile(root + os.sep + name, dst + os.sep + name)

            delete_folder(os.getcwd() + os.sep + repo.split('/')[-1].split('.')[0])


def load_data(path):
    location_data = {}
    twitter_data = {}
    dates = []
    for folder in os.listdir(path):
        if "twitter" in folder:
            for file in os.listdir(path + os.sep + folder):
                type = file.split("top1000")[-1].split('.')[0].replace("ii", "i")
                date = file.split('_')[0]
                if type not in twitter_data.keys():
                    twitter_data[type] = {}
                if date not in twitter_data[type].keys():
                    twitter_data[type][date] = []
                twitter_daily_data = load_csv_data(path + os.sep + folder + os.sep + file)
                twitter_data[type][date] = twitter_daily_data
                if date not in dates: dates.append(date)
        else:
            for file in os.listdir(path + os.sep + folder):
                date = file.split('.')[0]
                date = datetime.datetime.strptime(date, '%m-%d-%Y').strftime('%Y-%m-%d')
                if date not in location_data.keys(): location_data[date] = {}
                location_daily_data = load_csv_data(path + os.sep + folder + os.sep + file)
                location_data[date] = location_daily_data

    for date in list(location_data.keys()):
        if date not in set(dates):
            del location_data[date]

    del location_data["2020-03-22"]
    for type in list(twitter_data.keys()):
        del twitter_data[type]["2020-03-22"]

    return twitter_data, location_data


def to_data_frame(data):
    data_frame_list = []
    for date in list(data.keys()):
        daily_data = data[date]
        if "gram" not in daily_data[0] and "FIPS" not in daily_data[0]:
            header = ["gram", "count"]
        else:
            header = daily_data.pop(0)
        df = pd.DataFrame(daily_data, columns=header)
        df.insert(0, "Date", [date] * len(daily_data), True)
        data_frame_list.append(df)
    return data_frame_list


def df2db(db_name, table_name, df):
    db_connection = sqlite3.connect(db_name)
    if "location" in table_name.lower():
        data_frame = pd.concat(df, axis=0, join='outer', sort=False, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
    else:
        data_frame = pd.concat(df, axis=0, join='inner', sort=False, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
    data_frame.to_sql(table_name, db_connection, if_exists='replace', index=False)


if __name__ == "__main__":
    covid19_twitter_repo = r"https://github.com/thepanacealab/covid19_twitter.git"
    nCoV2019_CSSE_repo = r"https://github.com/CSSEGISandData/COVID-19.git"

    covid19_twitter_data_path = os.getcwd() + os.sep + covid19_twitter_repo.split('/')[-1].split('.')[0] + os.sep + "dailies"
    nCoV2019_location_data_path = os.getcwd() + os.sep + nCoV2019_CSSE_repo.split('/')[-1].split('.')[0] + os.sep + "csse_covid_19_data" + os.sep + "csse_covid_19_daily_reports"

    print("Checking COVID-19 GIT REPO data...")
    check_repo_data([(covid19_twitter_repo, covid19_twitter_data_path), (nCoV2019_CSSE_repo, nCoV2019_location_data_path)])

    print("Loading COVID-19 Twitter and Location data...")
    twitter_data, location_data = load_data(os.getcwd() + os.sep + "data")

    print("Creating Data Frames...")
    location_data_frame = to_data_frame(location_data)
    twitter_data_frame = {}
    for type in list(twitter_data.keys()):
        twitter_data_frame[type] = to_data_frame(twitter_data[type])

    print("Generating Sqlite DB...")
    df2db('covid19.db', "locations", location_data_frame)
    for type in list(twitter_data.keys()):
        df2db('covid19.db', type, twitter_data_frame[type])

    print("EOS")
