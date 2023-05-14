import urllib.request

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.data_cleansing import *


TRAIN_FILE_PATH = r'../dataset/ner_train_data.csv'
TEST_FILE_PATH = r'../dataset/ner_test_data.csv'
LABEL_FILE_PATH = r'../dataset/ner_label.txt'


def get_dataset_of_kakaobrain_nli(tokenizer=None, max_length_of_sequence=128):
    if tokenizer is None:
        raise Exception('Invalid tokenizer to parameter.')

    train_snli = pd.read_csv(TRAIN_SNLI_FILE_PATH, sep='\t', quoting=3)
    train_xnli = pd.read_csv(TRAIN_XNLI_FILE_PATH, sep='\t', quoting=3)
    test_data = pd.read_csv(TEST_FILE_PATH, sep='\t', quoting=3)
    val_data = pd.read_csv(VALIDATE_FILE_PATH, sep='\t', quoting=3)

    train_data = train_snli.append(train_xnli)
    train_data = train_data.sample(frac=1)

    train_data = drop_na_and_duplicates(train_data)
    test_data = drop_na_and_duplicates(test_data)
    val_data = drop_na_and_duplicates(val_data)

    X_train = convert_feature_by_nli(first_sentence_list=train_data['sentence1'],
                                     second_sentence_list=train_data['sentence2'],
                                     max_length_of_sequence=max_length_of_sequence,
                                     tokenizer=tokenizer)

    X_val = convert_feature_by_nli(first_sentence_list=val_data['sentence1'],
                                   second_sentence_list=val_data['sentence2'],
                                   max_length_of_sequence=max_length_of_sequence,
                                   tokenizer=tokenizer)

    X_test = convert_feature_by_nli(first_sentence_list=test_data['sentence1'],
                                    second_sentence_list=test_data['sentence2'],
                                    max_length_of_sequence=max_length_of_sequence,
                                    tokenizer=tokenizer)

    train_label = train_data['gold_label'].tolist()
    val_label = val_data['gold_label'].tolist()
    test_label = test_data['gold_label'].tolist()

    label_encoder = LabelEncoder()
    label_encoder.fit(train_label)

    y_train = label_encoder.transform(train_label)
    y_val = label_encoder.transform(val_label)
    y_test = label_encoder.transform(test_label)

    return X_train, y_train, X_test, y_test, X_val, y_val


def convert_feature_by_nli(first_sentence_list=None, second_sentence_list=None, max_length_of_sequence=None, tokenizer=None):
    input_id_list = list()
    attention_mask_list = list()
    token_type_id_list = list()

    for first_sentence, second_sentence in zip(first_sentence_list, second_sentence_list):
        encoding_result = tokenizer.encode_plus(first_sentence, second_sentence, max_length=max_length_of_sequence, pad_to_max_length=True)

        input_id_list.append(encoding_result['input_ids'])
        attention_mask_list.append(encoding_result['attention_mask'])
        token_type_id_list.append(encoding_result['token_type_ids'])

    input_id_list = np.array(input_id_list, dtype=int)
    attention_mask_list = np.array(attention_mask_list, dtype=int)
    token_type_id_list = np.array(token_type_id_list, dtype=int)

    return (input_id_list, attention_mask_list, token_type_id_list)


def download_ner_dataset(train_file_path=None, test_file_path=None, label_file_path=None):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/18.%20Fine-tuning%20BERT%20(Cls%2C%20NER%2C%20NLI)/dataset/ner_train_data.csv",
        filename=train_file_path)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/18.%20Fine-tuning%20BERT%20(Cls%2C%20NER%2C%20NLI)/dataset/ner_test_data.csv",
        filename=test_file_path)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/18.%20Fine-tuning%20BERT%20(Cls%2C%20NER%2C%20NLI)/dataset/ner_label.txt",
        filename=label_file_path)


if __name__ == '__main__':
    download_ner_dataset(train_file_path=TRAIN_FILE_PATH, test_file_path=TEST_FILE_PATH, label_file_path=LABEL_FILE_PATH)

    train_df = pd.read_csv(TRAIN_FILE_PATH)
    print(train_df.head())

    test_df = pd.read_csv(TEST_FILE_PATH)
    print(test_df.head())

