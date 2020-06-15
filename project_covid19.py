import os
import sys
import time
import csv
import paramiko
import subprocess
import zipfile
import tarfile
import shutil
from shutil import copytree, ignore_patterns
import numpy as np
import stat


def check_data(repos):
    if not os.path.exists(os.getcwd() + os.sep + "data"):

        for repo in repos: os.system("git clone " + repo)
        os.makedirs(os.getcwd() + os.sep + "data")

        for repo in repos:
            if "twitter" in repo:
                src = os.getcwd() + os.sep + repo.split('/')[-1].split('.')[0] + os.sep + "dailies"
                dst = os.getcwd() + os.sep + "data" + os.sep + repo.split('/')[-1].split('.')[0]
                copytree(src, dst, ignore=ignore_patterns("*.tsv.gz"))

                for root, dirs, files in os.walk(os.getcwd() + os.sep + covid19_twitter_repo.split('/')[-1].split('.')[0]):
                    for dir in dirs:
                        os.chmod(os.path.join(root, dir), stat.S_IRWXU)
                    for file in files:
                        os.chmod(os.path.join(root, file), stat.S_IRWXU)
                shutil.rmtree(os.getcwd() + os.sep + covid19_twitter_repo.split('/')[-1].split('.')[0], ignore_errors=True)


            else:
                src = os.getcwd() + os.sep + repo.split('/')[-1].split('.')[0] + os.sep + "latest_data"
                dst = os.getcwd() + os.sep + "data" + os.sep + repo.split('/')[-1].split('.')[0]
                copytree(src, dst)
                for file in os.listdir(dst):
                    if "tar" in file:
                        tar = tarfile.open(dst + os.sep + file, "r:gz")
                        tar.extractall(os.getcwd() + os.sep + "data" + os.sep + "nCoV2019")
                        tar.close()

                for root, dirs, files in os.walk(os.getcwd() + os.sep + nCoV2019_location_repo.split('/')[-1].split('.')[0]):
                    for dir in dirs:
                        os.chmod(os.path.join(root, dir), stat.S_IRWXU)
                    for file in files:
                        os.chmod(os.path.join(root, file), stat.S_IRWXU)
                shutil.rmtree(os.getcwd() + os.sep + nCoV2019_location_repo.split('/')[-1].split('.')[0], ignore_errors=True)


if __name__ == "__main__":
    covid19_twitter_repo = r"https://github.com/thepanacealab/covid19_twitter.git"
    nCoV2019_location_repo = r"https://github.com/beoutbreakprepared/nCoV2019.git"

    check_data([covid19_twitter_repo, nCoV2019_location_repo])
