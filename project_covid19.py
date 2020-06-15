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


def check_data(repos):
    if not os.path.exists(os.getcwd() + os.sep + "data"):

        for folder in os.listdir(os.getcwd() ):
            if os.path.isdir(folder) and ".idea" not in folder and ".git" not in folder:
                delete_folder(os.getcwd() + os.sep + folder)

        for repo in repos: os.system("git clone " + repo)
        os.makedirs(os.getcwd() + os.sep + "data")

        for repo in repos:
            if "twitter" in repo:
                src = os.getcwd() + os.sep + repo.split('/')[-1].split('.')[0] + os.sep + "dailies"
                dst = os.getcwd() + os.sep + "data" + os.sep + repo.split('/')[-1].split('.')[0]
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.tsv.gz"))
            else:
                src = os.getcwd() + os.sep + repo.split('/')[-1].split('.')[0] + os.sep + "latest_data"
                dst = os.getcwd() + os.sep + "data" + os.sep + repo.split('/')[-1].split('.')[0]
                shutil.copytree(src, dst)
                for file in os.listdir(dst):
                    if "tar" in file:
                        tar = tarfile.open(dst + os.sep + file, "r:gz")
                        tar.extractall(os.getcwd() + os.sep + "data" + os.sep + "nCoV2019")
                        tar.close()
            delete_folder(os.getcwd() + os.sep + repo.split('/')[-1].split('.')[0])


if __name__ == "__main__":
    covid19_twitter_repo = r"https://github.com/thepanacealab/covid19_twitter.git"
    nCoV2019_location_repo = r"https://github.com/beoutbreakprepared/nCoV2019.git"

    check_data([covid19_twitter_repo, nCoV2019_location_repo])
