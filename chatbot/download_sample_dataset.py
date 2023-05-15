import urllib.request
import re

import numpy as np
import pandas as pd


TRAIN_FILE_PATH = r'../dataset/songys_korean_chatbot.csv'


def get_dataset_of_korean_chatbot():
    train_data = pd.read_csv(TRAIN_FILE_PATH)

    question_list = list()
    for sentence in train_data['Q']:
        sentence = preprocess_sentence(sentence)

        question_list.append(sentence)

    answer_list = list()
    for sentence in train_data['A']:
        sentence = preprocess_sentence(sentence)

        answer_list.append(sentence)

    return question_list, answer_list


def preprocess_sentence(sentence):
    sentence = re.sub(r'([?.!,])', r" \1 ", sentence)
    sentence = sentence.strip()

    return sentence


def download_korean_chatbot_dataset(train_file_path=None):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
        filename=train_file_path)


if __name__ == '__main__':
    download_korean_chatbot_dataset(TRAIN_FILE_PATH)

    question_list, answer_list = get_dataset_of_korean_chatbot()

    print(question_list[:5])
    print(answer_list[:5])

