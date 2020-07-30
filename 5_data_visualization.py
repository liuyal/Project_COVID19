import os
import sys
import shutil
import stat
import collections
import imageio
import numpy as np
import pandas as pd
import wordcloud
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from PIL import Image


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


def plot_daily_token_count(cases_data, output_path):
    if not os.path.exists(os.sep.join(output_path.split(os.sep)[0:-1])):
        os.mkdir(os.sep.join(output_path.split(os.sep)[0:-1]))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        delete_folder(output_path)
        os.mkdir(output_path)

    for date in cases_data:
        words = list(cases_data[date])[0:10]
        count = [cases_data[date][word] for word in words]
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


def plot_daily_cases_count(location_data, output_path):
    if not os.path.exists(os.sep.join(output_path.split(os.sep)[0:-1])):
        os.mkdir(os.sep.join(output_path.split(os.sep)[0:-1]))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        delete_folder(output_path)
        os.mkdir(output_path)

    cases = []
    x = np.arange(0, len(location_data), 1)

    for date in location_data:
        cases.append(float(location_data[date]["confirmed"]))

        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots()
        fig.set_figheight(8)
        fig.set_figwidth(16)
        ax.set_xlim((0, 180))

        plt.ylim(0, 4000000)
        plt.plot(x[0:len(cases)], cases)
        plt.savefig(output_path + os.sep + date + '.png')
        plt.close()


def plot_combined(location_data, cases_data, output_path):
    if not os.path.exists(os.sep.join(output_path.split(os.sep)[0:-1])):
        os.mkdir(os.sep.join(output_path.split(os.sep)[0:-1]))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        delete_folder(output_path)
        os.mkdir(output_path)

    common_dates = list(set(cases_data.keys()).intersection(set(location_data.keys())))
    common_dates.sort()
    cases = []
    x = np.arange(0, len(location_data), 1)

    for date in common_dates:
        cases.append(float(location_data[date]["confirmed"]))
        words = list(cases_data[date])[0:6]
        count = [cases_data[date][word] for word in words]
        y_pos = np.arange(len(words))

        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(2)
        fig.set_figheight(9)
        fig.set_figwidth(16)
        fig.suptitle(date, fontsize=22, y=0.95)

        ax[0].set_xticks([])
        ax[0].set_xlim((0, 180))
        ax[0].set_ylim((0, 4000000))
        ax[0].set_ylabel("Number of Confirmed Cases (US)", labelpad=20)
        ax[0].plot(x[0:len(cases)], cases, color='#9E1A1A', linewidth=5)
        ax[0].plot(x[len(cases) - 1], cases[-1], color='#9E1A1A', marker='o', markerfacecolor='#9E1A1A', markersize=8)

        ax[1].set_xlim((0, 600))
        ax[1].set_yticks(y_pos)
        ax[1].set_yticklabels(words)
        ax[1].invert_yaxis()
        ax[1].barh(y_pos, count, align='center')

        plt.savefig(output_path + os.sep + date + '.png')
        plt.close()


def make_gif(image_folder_path, output_path):
    images = []
    for image_name in os.listdir(image_folder_path):
        images.append(imageio.imread(image_folder_path + os.sep + image_name))
    imageio.mimsave(output_path, images, fps=7)


def tweet_wordcount_frequency_distribution(input_path, output_path):
    wordcount_daily = {}
    wordcount_list = []
    for file_name in os.listdir(input_path):
        if "count" in file_name:
            date = file_name.split('_')[0]
            file = open(input_path + os.sep + file_name)
            lines = file.readlines()
            lines.pop(0)
            file.close()
            wordcount_daily[date] = []
            item_list = {}
            for item in lines:
                word = item.replace('\n', '').split(',')[0]
                count = item.replace('\n', '').split(',')[1]
                item_list[word] = int(count)
            wordcount_daily[date].append(collections.Counter(item_list))
            wordcount_list.append(collections.Counter(item_list))
    wordcount_total = sum(wordcount_list, collections.Counter())

    file = open(output_path, "a+")
    file.truncate(0)
    file.write("word,count\n")
    for word, count in wordcount_total.most_common():
        file.write(word + "," + str(count) + '\n')
        file.flush()
    file.close()

    return wordcount_daily, wordcount_total


# TODO: Plot sentiment count
def plot_tweet_sentiment(input_path, output_path):
    if not os.path.exists(os.sep.join(output_path.split(os.sep)[0:-1])):
        os.mkdir(os.sep.join(output_path.split(os.sep)[0:-1]))


def red_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    lmin = 20.0
    lmax = 65.0
    hue = 10
    saturation = 100
    luminance = int((lmax - lmin) * (float(font_size) / 500.0) + lmin)
    return "hsl({}, {}%, {}%)".format(hue, saturation, luminance)


def tweet_word_cloud_maker(word_count, color_func, output_path):
    if not os.path.exists(os.sep.join(output_path.split(os.sep)[0:-1])):
        os.mkdir(os.sep.join(output_path.split(os.sep)[0:-1]))
    word_cloud = wordcloud.WordCloud(width=1600, height=900)
    word_cloud.max_font_size = 500
    word_cloud.min_font_size = 15
    word_cloud.background_color = "white"
    word_cloud.color_func = color_func
    wc_image = word_cloud.generate_from_frequencies(word_count)
    plt.figure(figsize=(12, 12), facecolor=None)
    plt.imshow(wc_image, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', transparent=True)
    image = Image.open(output_path)
    # image.show()


# TODO: Plot word count distribution
def tweet_word_cloud_distribution_plotter(word_count, output_path):
    if not os.path.exists(os.sep.join(output_path.split(os.sep)[0:-1])):
        os.mkdir(os.sep.join(output_path.split(os.sep)[0:-1]))


if __name__ == "__main__":
    daily_us_cases_data_results_path = os.getcwd() + os.sep + "data" + os.sep + "daily_us_confirmed_cases.csv"
    tweet_tokenized_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_tokenized_tweets"

    word_count_chart_image_folder = os.getcwd() + os.sep + "data" + os.sep + "images" + os.sep + "plot_word_count"
    word_count_chart_gif_path = os.getcwd() + os.sep + "data" + os.sep + "images" + os.sep + "plot_word_count.gif"
    confirmed_cases_chart_image_folder = os.getcwd() + os.sep + "data" + os.sep + "images" + os.sep + "plot_confirmed_cases"
    confirmed_cases_chart_gif_path = os.getcwd() + os.sep + "data" + os.sep + "images" + os.sep + "plot_confirmed_cases.gif"
    combined_chart_image_folder = os.getcwd() + os.sep + "data" + os.sep + "images" + os.sep + "plot_combined"
    combined_chart_gif_path = os.getcwd() + os.sep + "data" + os.sep + "images" + os.sep + "plot_combined.gif"

    tweet_sentiment_data_directory = os.getcwd() + os.sep + "data" + os.sep + "tweet_sentiment_result.csv"
    tweet_token_distribution_directory = os.getcwd() + os.sep + "data" + os.sep + "tweet_token_distribution.csv"

    tweet_sentiment_output_path = os.getcwd() + os.sep + "data" + os.sep + "tweet_sentiment.png"
    tweet_word_cloud_output_path = os.getcwd() + os.sep + "data" + os.sep + "images" + os.sep + "tweet_word_cloud.png"
    tweet_wc_distribution_output_path = os.getcwd() + os.sep + "data" + os.sep + "images" + os.sep + "tweet_word_distribution.png"

    print("Running Data Visualizer...")
    cases_count_daily = load_cases_count(daily_us_cases_data_results_path)
    token_count_daily = load_tweet_token_count(tweet_tokenized_directory)
    
    print("Plotting daily top token count...")
    plot_daily_token_count(token_count_daily, word_count_chart_image_folder)
    make_gif(word_count_chart_image_folder, word_count_chart_gif_path)
    
    print("Plotting daily number of confirmed cases US...")
    plot_daily_cases_count(cases_count_daily, confirmed_cases_chart_image_folder)
    make_gif(confirmed_cases_chart_image_folder, confirmed_cases_chart_gif_path)
    
    print("Plotting combined token/cases chart...")
    plot_combined(cases_count_daily, token_count_daily, combined_chart_image_folder)
    make_gif(combined_chart_image_folder, combined_chart_gif_path)

    print("Plotting Tweet sentiment count...")
    plot_tweet_sentiment(tweet_sentiment_data_directory, tweet_sentiment_output_path)

    print("Generating Word Cloud...")
    word_count_daily, word_count_total = tweet_wordcount_frequency_distribution(tweet_tokenized_directory, tweet_token_distribution_directory)
    tweet_word_cloud_maker(word_count_total, red_color_func, tweet_word_cloud_output_path)

    print("Plotting word count distribution...")
    tweet_word_cloud_distribution_plotter(word_count_total, tweet_wc_distribution_output_path)

    print("Visualization Complete!")
