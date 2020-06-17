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
import csv
import subprocess
import tarfile
import shutil
import numpy
import stat
import PIL


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
    for row in data: data_output.append(row)
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
        else:
            for file in os.listdir(path + os.sep + folder):
                date = file.split('.')[0]
                if date not in location_data.keys(): location_data[date] = {}
                location_daily_data = load_csv_data(path + os.sep + folder + os.sep + file)
                location_data[date] = location_daily_data

    return twitter_data, location_data


if __name__ == "__main__":
    covid19_twitter_repo = r"https://github.com/thepanacealab/covid19_twitter.git"
    nCoV2019_location_repo = r"https://github.com/CSSEGISandData/COVID-19.git"

    covid19_twitter_data_path = os.getcwd() + os.sep + covid19_twitter_repo.split('/')[-1].split('.')[0] + os.sep + "dailies"
    nCoV2019_location_data_path = os.getcwd() + os.sep + nCoV2019_location_repo.split('/')[-1].split('.')[0] + os.sep + "csse_covid_19_data" + os.sep + "csse_covid_19_daily_reports"

    print("Checking COVID-19 GIT REPO data...")
    check_repo_data([(covid19_twitter_repo, covid19_twitter_data_path), (nCoV2019_location_repo, nCoV2019_location_data_path)])

    print("Loading COVID-19 Twitter and Location data...")
    twitter_data, location_data = load_data(os.getcwd() + os.sep + "data")

    print("EOS")
