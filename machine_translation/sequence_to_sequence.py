import os
import shutil
import zipfile
import urllib3

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model

# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/device:GPU:0"])

TRAIN_FILE_PATH = '../dataset/fra.txt'


def get_char_vocab_list(lines):
    vocab = set()
    for line in lines:
        for char in line:
            vocab.add(char)

    return sorted(list(vocab))


def integer_encoding(line, index_dict):
    encoded_line = list()
    for char in line:
        encoded_line.append(index_dict[char])

    return encoded_line


def decode_sequence(encoder_model=None, input_sequence=None, target_vocal_size=None, index_dict_to_target=None, max_length_by_target=None):
    states_value = encoder_model.predict(input_sequence)

    target_sequence = np.zeros((1, 1, target_vocal_size))
    target_sequence[0, 0, index_dict_to_target['\t']] = 1.

    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_sequence] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_tar[sampled_token_index]

        decoded_sentence += sampled_char

        if (sampled_char == '\n') or (len(decoded_sentence) > max_length_by_target):
            stop_condition = True

        target_sequence = np.zeros((1, 1, target_vocal_size))
        target_sequence[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence


if __name__ == '__main__':
    lines = pd.read_csv(TRAIN_FILE_PATH, names=['src', 'tar', 'lic'], sep='\t')
    lines = lines[0:35000]
    del lines['lic']

    lines.tar = lines.tar.apply(lambda x : '\t ' + x + ' \n')

    source_vocab = get_char_vocab_list(lines.src)
    target_vocab = get_char_vocab_list(lines.tar)

    index_dict_to_source = dict([(word, i + 1) for i, word in enumerate(source_vocab)])
    index_dict_to_target = dict([(word, i + 1) for i, word in enumerate(target_vocab)])

    encoder_input_list = list()
    for line in lines.src:
        encoder_input_list.append(integer_encoding(line, index_dict_to_source))

    decoder_input_list = list()
    for line in lines.tar:
        decoder_input_list.append(integer_encoding(line, index_dict_to_target))

    decoder_target_list = list()
    for line in lines.tar:
        decoder_target_list.append(integer_encoding(line[2:], index_dict_to_target))

    max_length_by_source = max([len(line) for line in lines.src])
    max_length_by_target = max([len(line) for line in lines.tar])

    encoder_input_list = pad_sequences(encoder_input_list, maxlen=max_length_by_source, padding='post')
    decoder_input_list = pad_sequences(decoder_input_list, maxlen=max_length_by_target, padding='post')
    decoder_target_list = pad_sequences(decoder_target_list, maxlen=max_length_by_target, padding='post')

    encoder_input_list = to_categorical(encoder_input_list)
    decoder_input_list = to_categorical(decoder_input_list)
    decoder_target_list = to_categorical(decoder_target_list)

    with tf.device('/device:GPU:0'):
        encoder_inputs = Input(shape=(None, len(source_vocab) + 1))
        encoder_lstm = LSTM(units=256, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, len(target_vocab) + 1))
        decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        decoder_softmax_layer = Dense(len(target_vocab) + 1, activation='softmax')
        decoder_outputs = decoder_softmax_layer(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.fit(x=[encoder_input_list, decoder_input_list], y=decoder_target_list, batch_size=64, epochs=20, validation_split=0.2)

    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

    decoder_state_input_h = Input(shape=(256,))
    decoder_state_input_c = Input(shape=(256,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_softmax_layer(decoder_outputs)
    decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)

    index_to_src = dict((i, char) for char, i in index_dict_to_source.items())
    index_to_tar = dict((i, char) for char, i in index_dict_to_target.items())

    for sequence_index in [3, 50, 100, 300, 1001]:
        input_sequence = encoder_input_list[sequence_index : sequence_index + 1]
        decoded_sentence = decode_sequence(encoder_model=encoder_model,
                                           input_sequence=input_sequence,
                                           target_vocal_size=len(target_vocab) + 1,
                                           index_dict_to_target=index_dict_to_target,
                                           max_length_by_target=max_length_by_target)

        print(35 * "-")
        print('입력 문장:', lines.src[sequence_index])
        print('정답 문장:', lines.tar[sequence_index][2:len(lines.tar[sequence_index]) - 1])
        print('번역 문장:', decoded_sentence[1:len(decoded_sentence) - 1])
