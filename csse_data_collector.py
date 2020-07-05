# ----------------------------------------------------------------------
# DATE: 2020/08/10
# AUTHOR: Jerry Liu
# EMAIL: Liuyal@sfu.ca
#
# DESCRIPTION:
# CSSE confirmed cases data collector
# ----------------------------------------------------------------------

import os
import sys
import time
import datetime
import csv
import shutil
import stat
import sqlite3
import requests
import pandas as pd


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


def check_repo_data(dst, url, start_date="2020-03-22"):
    if not os.path.exists(dst):
        for folder in os.listdir(os.getcwd()):
            if os.path.isdir(folder) and ".idea" not in folder and ".git" not in folder and "data" not in folder:
                delete_folder(os.getcwd() + os.sep + folder)
        os.makedirs(os.getcwd() + os.sep + "data")
        if not os.path.isdir(dst): os.mkdir(dst)
        response = requests.get(url)
        text_list = response.text.split('\n')
        csv_list = []
        for item in text_list:
            if "js-navigation-open link-gray-dark" in item and ".csv" in item:
                csv_url = "https://raw.githubusercontent.com" + item[item.index('href="') + len('href="'): item.index('"', item.index('href="') + len('href="'), -1)]
                csv_list.append(csv_url.replace("blob/", ""))
        for item in csv_list:
            date = datetime.datetime.strptime(item.split('/')[-1].split('.')[0], '%m-%d-%Y').strftime('%Y-%m-%d')
            if date > start_date:
                response = requests.get(item)
                f = open(dst + os.sep + date + ".csv", "w", encoding="utf-8")
                f.write(response.text)
                f.flush()
                f.close()

    elif len(os.listdir(dst)) > 0:
        dates = [file.replace(".csv", "") for file in os.listdir(dst) if ".csv" in file]
        response = requests.get(url)
        text_list = response.text.split('\n')
        csv_list = []
        for item in text_list:
            if "js-navigation-open link-gray-dark" in item and ".csv" in item:
                csv_url = "https://raw.githubusercontent.com" + item[item.index('href="') + len('href="'): item.index('"', item.index('href="') + len('href="'), -1)]
                csv_list.append(csv_url.replace("blob/", ""))
        for item in csv_list:
            date = datetime.datetime.strptime(item.split('/')[-1].split('.')[0], '%m-%d-%Y').strftime('%Y-%m-%d')
            if date > max(dates):
                response = requests.get(item)
                f = open(dst + os.sep + date + ".csv", "w", encoding="utf-8")
                f.write(response.text)
                f.flush()
                f.close()


def load_data(path, start_date="2020-03-22"):
    location_data = {}
    for file in os.listdir(path):
        date = file.split('.')[0]
        if date > start_date:
            if date not in location_data.keys(): location_data[date] = {}
            location_daily_data = load_csv_data(path + os.sep + file)
            location_data[date] = location_daily_data
    return location_data


def to_data_frame(data):
    data_frame_list = []
    for date in list(data.keys()):
        daily_data = data[date]
        header = daily_data.pop(0)
        df = pd.DataFrame(daily_data, columns=header)
        df.insert(0, "Date", [date] * len(daily_data), True)
        data_frame_list.append(df)
    return data_frame_list


def df2db(db_name, table_name, df):
    db_connection = sqlite3.connect(db_name)
    data_frame = pd.concat(df, axis=0, join='outer', sort=False, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
    data_frame.to_sql(table_name, db_connection, if_exists='replace', index=False)


if __name__ == "__main__":
    nCoV2019_CSSE_repo = r"https://github.com/CSSEGISandData/COVID-19.git"
    nCoV2019_CSSE_data_url = r"https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports"
    nCoV2019_CSSE_data_path = os.getcwd() + os.sep + "data" + os.sep + nCoV2019_CSSE_repo.split('/')[-1].split('.')[0].lower().replace('-', '_') + "_location_data"

    print("Checking COVID-19 GIT REPO data...")
    check_repo_data(nCoV2019_CSSE_data_path, nCoV2019_CSSE_data_url)

    print("Loading COVID-19 Locations data...")
    location_data = load_data(nCoV2019_CSSE_data_path, "2020-03-22")

    print("Creating COVID-19 Locations Data Frames...")
    location_data_frame = to_data_frame(location_data)

    print("Generating COVID-19 Locations Sqlite DB...")
    df2db("data" + os.sep + "covid19.db", "locations", location_data_frame)
