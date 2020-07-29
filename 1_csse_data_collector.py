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


def check_repo_data(dst, url, start_date="2020-03-22"):
    if not os.path.exists(dst):
        for folder in os.listdir(os.getcwd()):
            if os.path.isdir(folder) and ".idea" not in folder and ".git" not in folder and "data" not in folder:
                delete_folder(os.getcwd() + os.sep + folder)
        if not os.path.exists(os.getcwd() + os.sep + "data"): os.makedirs(os.getcwd() + os.sep + "data")
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
            if date >= start_date:
                response = requests.get(item)
                f = open(dst + os.sep + date + ".csv", "w", encoding="utf-8")
                f.write(response.text.replace("\r\n", "\n"))
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
            if date >= max(dates):
                response = requests.get(item)
                f = open(dst + os.sep + date + ".csv", "w", encoding="utf-8")
                f.write(response.text.replace("\r\n", "\n"))
                f.flush()
                f.close()


def load_csv_data(file_path):
    data_output = []
    file = open(file_path, "r+", encoding="utf-8")
    data = csv.reader(file)
    for row in data:
        for i in range(0, len(row)):
            if row[i].isnumeric():
                row[i] = float(row[i])
            elif row[i] == "":
                row[i] = 0.0
            if 'Country/Region' == row[i]:
                row[i] = 'Country_Region'
        data_output.append(row)
    file.close()
    return data_output


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
    column_keywords = ['Date', 'Country_Region', 'Confirmed', 'Deaths', 'Recovered']
    data_frame_list = []
    for date in list(data.keys()):
        daily_data = data[date]
        header = daily_data.pop(0)
        df = pd.DataFrame(daily_data, columns=header)
        df.insert(0, "Date", [date] * len(daily_data), True)
        drop_list = []
        for item in df.columns:
            if item not in column_keywords:
                drop_list.append(item)
        df.drop(drop_list, axis=1, inplace=True)
        data_frame_list.append(df)
    return data_frame_list


def process_daily_cases_us(data_frame, file_path):
    df = pd.concat(data_frame, axis=0, join='outer', sort=False, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
    daily_us = df.loc[df["Country_Region"] == "US"]
    aggregate = {"Confirmed": "sum", "Deaths": "sum", "Recovered": "sum"}
    daily_us_total = daily_us.groupby("Date").agg(aggregate).reset_index()
    daily_us_total.to_csv(file_path)
    return daily_us_total


def process_daily_cases_global(data_frame, file_path):
    df = pd.concat(data_frame, axis=0, join='outer', sort=False, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
    aggregate = {"Confirmed": "sum", "Deaths": "sum", "Recovered": "sum"}
    daily_total = df.groupby("Date").agg(aggregate).reset_index()
    daily_total.to_csv(file_path)
    return daily_total


def df2db(df, table_name, db_name):
    db_connection = sqlite3.connect(db_name)
    data_frame = pd.concat(df, axis=0, join='outer', sort=False, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
    data_frame.to_sql(table_name, db_connection, if_exists='replace', index=False)


if __name__ == "__main__":
    nCoV2019_CSSE_repo = r"https://github.com/CSSEGISandData/COVID-19.git"
    nCoV2019_CSSE_data_url = r"https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports"
    nCoV2019_CSSE_data_path = os.getcwd() + os.sep + "data" + os.sep + nCoV2019_CSSE_repo.split('/')[-1].split('.')[0].lower().replace('-', '_') + "_location_data"
    daily_us_cases_results_path = os.getcwd() + os.sep + "data" + os.sep + "daily_us_confirmed_cases.csv"
    daily_global_cases_results_path = os.getcwd() + os.sep + "data" + os.sep + "daily_global_confirmed_cases.csv"
    db_path = os.getcwd() + os.sep + "data" + os.sep + "covid19_csse_database.db"

    print("Checking COVID-19 GIT REPO data...")
    check_repo_data(nCoV2019_CSSE_data_path, nCoV2019_CSSE_data_url, "2020-01-01")

    print("Loading COVID-19 CSSE data...")
    csse_data = load_data(nCoV2019_CSSE_data_path, "2020-01-01")

    print("Creating COVID-19 CSSE Data Frames...")
    csse_data_frame = to_data_frame(csse_data)

    print("Processing COVID-19 CSSE Data Frames...")
    process_daily_cases_us(csse_data_frame, daily_us_cases_results_path)
    process_daily_cases_global(csse_data_frame, daily_global_cases_results_path)

    print("Generating COVID-19 Confirmed Cases Sqlite DB...")
    df2db(csse_data_frame, "cases", db_path)

    print("CSSE Data Collection Complete!")
