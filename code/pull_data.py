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


def ssh_connect(hostname, username, password, port):
    client = paramiko.client.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=hostname, username=username, password=password, port=port)
    stdin, stdout, stderr = client.exec_command("hostnamectl")
    if len(stderr.readlines()) == 0:
        print("".join(stdout.readlines()[0:-1]))
    else:
        print("".join(stderr.readlines()))
    return client


def get_repo(path, repos, client):
    stdin, stdout, stderr = client.exec_command("cd " + path + "; ls -l")
    response = stdout.readlines()
    print("".join(response).replace("\n\n", "\n"))

    data_exist = False
    for item in response:
        if "data" in item:
            data_exist = True
            break

    if not data_exist:
        client.exec_command("cd " + path + "; mkdir data")
        for repo in repos:
            cmd = "cd " + path + "/data" + "; git clone " + repo
            stdin, stdout, stderr = client.exec_command(cmd)
            response = stdout.readlines()
            print("".join(response).replace("\n\n", "\n"))


def git_repo_check(path, client):
    stdin, stdout, stderr = client.exec_command("cd " + path + "; git status")
    response = stdout.readlines()
    print("".join(response).replace("\n\n", "\n"))
    if "nothing to commit" not in response[-1]:
        stdin, stdout, stderr = client.exec_command("cd " + path + "; git pull")
        print("".join(stdout.readlines()).replace("\n\n", "\n"))


def get_location_data(path):
    if not os.path.exists(os.getcwd() + os.sep + "data" + os.sep + "nCoV2019"):
        os.makedirs(os.getcwd() + os.sep + "data" + os.sep + "nCoV2019")
    location_files = os.listdir(path)
    for file in location_files:
        if "tar" in file:
            tar = tarfile.open(path + os.sep + file, "r:gz")
            tar.extractall(os.getcwd() + os.sep + "data" + os.sep + "nCoV2019")
            tar.close()


def get_twitter_data(path):
    copytree(path, os.getcwd() + os.sep + "data" + os.sep + "covid19_twitter", ignore=ignore_patterns("*.tsv.gz"))


if __name__ == "__main__":
    hostname = "192.168.1.100"
    username = "root"
    password = "1234"
    port = 22

    covid19_twitter_repo = r"https://github.com/thepanacealab/covid19_twitter.git"
    nCoV2019_location_repo = r"https://github.com/beoutbreakprepared/nCoV2019.git"
    directory = "~/Desktop/share"

    client = ssh_connect(hostname, username, password, port)

    get_repo(directory, [covid19_twitter_repo, nCoV2019_location_repo], client)
    git_repo_check(directory + "/data/covid19_twitter", client)
    git_repo_check(directory + "/data/nCoV2019", client)

    shutil.rmtree(os.getcwd() + os.sep + "data")
    os.makedirs(os.getcwd() + os.sep + "data")

    get_location_data(r"\\" + hostname + r"\share\data\nCoV2019\latest_data")
    get_twitter_data(r"\\" + hostname + r"\share\data\covid19_twitter\dailies")
