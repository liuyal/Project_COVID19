import os
import sys
import time
import csv
import paramiko
import subprocess
import shutil
import numpy as np
from shutil import copytree, ignore_patterns


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


if __name__ == "__main__":
    hostname = "192.168.1.100"
    username = "root"
    password = "1234"
    port = 22

    directory = "~/Desktop/share"
    covid19_twitter_repo = r"https://github.com/thepanacealab/covid19_twitter.git"
    covid19_twitter_id_repo = r"https://github.com/echen102/COVID-19-TweetIDs.git"
    nCoV2019_location_repo = r"https://github.com/CSSEGISandData/COVID-19.git"

    client = ssh_connect(hostname, username, password, port)

    get_repo(directory, [covid19_twitter_repo, covid19_twitter_id_repo, nCoV2019_location_repo], client)
    git_repo_check(directory + "/data/covid19_twitter", client)
    git_repo_check(directory + "/data/COVID-19", client)
    git_repo_check(directory + "/data/COVID-19-TweetIDs", client)
