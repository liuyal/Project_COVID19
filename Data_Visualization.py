import os
import sys
import shutil
import stat
import collections
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML


def delete_folder(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.chmod(os.path.join(root, dir), stat.S_IRWXU)
        for file in files:
            os.chmod(os.path.join(root, file), stat.S_IRWXU)
    shutil.rmtree(path, ignore_errors=True)


def load_cases_count(file_path):
    file = open(file_path, "r+")
    text = file.readlines()
    text.pop(0)
    file.close()
    data = {}
    for line in text:
        data[line.replace('\n', '').split(',')[1]] = {}
        data[line.replace('\n', '').split(',')[1]]["confirmed"] = line.replace('\n', '').split(',')[2]
        data[line.replace('\n', '').split(',')[1]]["deaths"] = line.replace('\n', '').split(',')[3]
        data[line.replace('\n', '').split(',')[1]]["recovered"] = line.replace('\n', '').split(',')[4]
    return data


def load_tweet_token_count(input_path):
    wordcount_daily = {}
    for file_name in os.listdir(input_path):
        if "count" in file_name:
            date = file_name.split('_')[0]
            file = open(input_path + os.sep + file_name)
            lines = file.readlines()
            lines.pop(0)
            file.close()

            item_list = {}
            for item in lines:
                word = item.replace('\n', '').split(',')[0]
                count = item.replace('\n', '').split(',')[1]
                item_list[word] = int(count)
            wordcount_daily[date] = (collections.Counter(item_list).most_common())

    for date in wordcount_daily:
        temp_data = {}
        for word, count in wordcount_daily[date]:
            temp_data[word] = count
        wordcount_daily[date] = collections.Counter(temp_data)

    return wordcount_daily


def format_data(cases_count_daily, token_count_daily, output_path):
    common_dates = list(set(cases_count_daily.keys()).intersection(set(token_count_daily.keys())))
    common_dates.sort()
    word_sets = []
    for date in common_dates:
        for word in token_count_daily[date]:
            word_sets.append(word)
    word_sets = list(set(word_sets))
    word_sets.sort()

    data_list = []
    for date in common_dates:
        values = []
        for word in word_sets:
            if word in token_count_daily[date]:
                values.append(token_count_daily[date][word])
            else:
                values.append(0)
        data_list.append(date + "," + ",".join([str(x) for x in values]))

    file = open(output_path, "a+")
    file.truncate(0)
    file.write("date," + ",".join(word_sets) + '\n')
    file.write("\n".join(data_list) + '\n')
    file.flush()
    file.close()


def plot_bar_race(data, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        delete_folder(output_path)
        os.mkdir(output_path)

    for date in data:
        print(date, data[date].most_common(10))
        words = list(data[date])[0:10]
        count = [data[date][word] for word in words]
        y_pos = np.arange(len(words))

        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots()
        fig.set_figheight(8)
        fig.set_figwidth(16)
        fig.suptitle(date, fontsize=22)

        ax.barh(y_pos, count, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlim([0, 600])

        plt.savefig(output_path + os.sep + date + '.png')
        plt.close()

def make_gif(image_folder_path, output_path):
    images = []
    for image_name in os.listdir(image_folder_path):
        images.append(imageio.imread(image_folder_path + os.sep + image_name))
    imageio.mimsave(output_path, images, fps=7)


if __name__ == "__main__":
    location_data_results_path = os.getcwd() + os.sep + "data" + os.sep + "daily_us_confirmed_cases.csv"
    tweet_tokenized_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_tokenized_tweets"
    combined_data_path = os.getcwd() + os.sep + "data" + os.sep + "bar_chart_race.csv"
    image_folder = os.getcwd() + os.sep + "data" + os.sep + "barcharts"
    bar_chart_gif_path = os.getcwd() + os.sep + "data" + os.sep + "barcharts.gif"

    cases_count_daily = load_cases_count(location_data_results_path)
    token_count_daily = load_tweet_token_count(tweet_tokenized_directory)
    # format_data(cases_count_daily, token_count_daily, combined_data_path)

    # plot_bar_race(token_count_daily, image_folder)
    make_gif(image_folder, bar_chart_gif_path)

    print("EOS")
