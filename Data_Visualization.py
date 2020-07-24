import os
import sys
import collections
import bar_chart_race as bcr


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


def combine_data(cases_count_daily, token_count_daily):
    combined_data = {}
    common_dates = list(set(cases_count_daily.keys()).intersection(set(token_count_daily.keys())))
    common_dates.sort()

    word_sets = []
    for date in common_dates:
        for word in token_count_daily[date]:
            word_sets.append(word)
    word_sets = set(word_sets)

    for date in common_dates:
        combined_data[date] = {}

        # combined_data[date]["cases_confirmed_count"] = cases_count_daily[date]["confirmed"]
        # combined_data[date]["cases_deaths_count"] = cases_count_daily[date]["deaths"]
        # combined_data[date]["cases_recovered_count"] = cases_count_daily[date]["recovered"]

        for word in word_sets:
            if word in token_count_daily[date]:
                combined_data[date][word] = token_count_daily[date][word]
            else:
                combined_data[date][word] = 0

    return combined_data


def save_combined_data(combined_data, output_path):
    file = open(output_path, "a+")
    file.truncate(0)

    header = list(combined_data.keys())
    header.sort()
    file.write("words," + ",".join(header) + "\n")
    file.flush()

    words = sorted(list(combined_data[header[0]]))

    for word in words:
        values = []
        for date in header:
            values.append(combined_data[date][word])
        file.write(word + "," + ",".join([str(x) for x in values]) + '\n')
        file.flush()
    file.close()



if __name__ == "__main__":
    location_data_results_path = os.getcwd() + os.sep + "data" + os.sep + "daily_us_confirmed_cases.csv"
    tweet_tokenized_directory = os.getcwd() + os.sep + "data" + os.sep + "covid_19_tokenized_tweets"
    combined_data_path = os.getcwd() + os.sep + "data" + os.sep + "bar_chart_race.csv"

    cases_count_daily = load_cases_count(location_data_results_path)
    token_count_daily = load_tweet_token_count(tweet_tokenized_directory)
    combined_data = combine_data(cases_count_daily, token_count_daily)
    save_combined_data(combined_data, combined_data_path)

    print("EOS")
