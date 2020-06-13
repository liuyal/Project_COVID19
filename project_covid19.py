import os
import sys
import time
import csv
import paramiko


def ssh_connect(hostname, username, password, port):
    client = paramiko.client.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=hostname, username=username, password=password, port=port)
    stdin, stdout, stderr = client.exec_command("lscpu")
    print(stdout.readlines())
    print(stderr.readlines())


if __name__ == "__main__":

    hostname = "192.168.1.100"
    username = "root"
    password = "1234"
    port = 22

    location_data_path = "\\\\192.168.1.100\share\Data\location\latest_data"
    twitter_data_path = "\\\\192.168.1.100\share\Data\\twitter\dailies"
    print(os.listdir(location_data_path))
    print(os.listdir(twitter_data_path))

    ssh_connect(hostname, username, password, port)