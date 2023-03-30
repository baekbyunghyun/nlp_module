import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request

from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import dump_svmlight_file
from tensorflow.keras.


def get_vocab_size(tokenizer, threshold=3):
    total_cnt = len(tokenizer.word_index)
    rare_cnt = 0
    total_freq = 0
    rare_freq = 0

    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        if value < threshold:
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print('단어 집합(vocabulary)의 크기 :', total_cnt)
    print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
    print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

    return total_cnt - rare_cnt + 1


if __name__ == '__main__':
    train_data = pd.read_table('dataset/ratings_train.txt')
    test_data = pd.read_table('dataset/ratings_test.txt')

    # print(train_data['document'].nunique())
    # print(train_data['label'].nunique())

    train_data.drop_duplicates(subset=['document'], inplace=True)
    print('총 샘플의 수 :', len(train_data))

    # print(train_data.isnull().values.any())
    # print(train_data.isnull().sum())

    train_data = train_data.dropna(how="any")
    train_data['document'] = train_data['document'].str.replace("[^ ㄱ-ㅎㅏ-ㅣ가-힣]", "")
    train_data['document'] = train_data['document'].str.replace('^ +', "")
    train_data['document'].replace('', np.nan, inplace=True)
    train_data = train_data.dropna(how="any")

    test_data.drop_duplicates(subset=['document'], inplace=True)
    test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    test_data['document'] = test_data['document'].str.replace('^ +', "")
    test_data['document'].replace('', np.nan, inplace=True)
    test_data = test_data.dropna(how='any')

    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    okt = Okt()

    X_train = list()
    for sentence in tqdm(train_data['document']):
        tokenized_sentence = okt.morphs(sentence, stem=True)
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]

        X_train.append(stopwords_removed_sentence)

    X_test = list()
    for sentence in tqdm(test_data['document']):
        tokenized_sentence = okt.morphs(sentence, stem=True)
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]

        X_test.append(stopwords_removed_sentence)

    print(X_train[:3])
    print(X_test[:3])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    vocal_size = get_vocab_size(tokenizer, threshold=3)

    tokenizer = Tokenizer(vocal_size)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])

    # drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
    #
    # X_train = np.delete(X_train, drop_train, axis=0)
    # y_train = np.delete(y_train, drop_train, axis=0)

    X_train = pad_sequences(X_train, maxlen=30)
    X_test = pad_sequences(X_test, maxlen=30)

    print(X_train[:5])
    print(y_train[:5])

    dump_svmlight_file(X_train, y_train, "dataset/train_naver_movie_review.spa", zero_based=False)
    dump_svmlight_file(X_test, y_test, "dataset/test_naver_movie_review.spa", zero_based=False)

