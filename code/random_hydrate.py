# ----------------------------------------------------------------------
# DATE: 2020/08/10
# AUTHOR: Jerry Liu
# EMAIL: Liuyal@sfu.ca
#
# DESCRIPTION:
# Tweet hydrator daily 1000 random
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
import paramiko
import gzip
import json
import random
from tqdm import tqdm
from twarc import Twarc
from pathlib import Path


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


def collect_tweet_id(directory):
    data_files = []
    for item in os.listdir(directory):
        if os.path.isdir(directory + os.sep + item) and "-" in item:
            for file in os.listdir(directory + os.sep + item):
                data_files.append(directory + os.sep + item + os.sep + file)

    id_log = {}
    for file in data_files:
        # date = '-'.join(file.split(os.sep)[-1].split('.')[0].split('id-')[-1].split('-')[0:-1])
        f = open(file, "r")
        id_list = f.read()
        f.close()
        # if date not in list(id_log.keys()): id_log[date] = []
        # id_log[date] = id_log[date] + id_list
        print(file)



if __name__ == "__main__":
    data_directory = ""

    collect_tweet_id(data_directory)

    print("\nEOS")
