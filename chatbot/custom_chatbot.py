import time

import tensorflow as tf
import tensorflow_datasets as tfds

from chatbot.download_sample_dataset import *
from custom_transformer.transformer import *
from custom_transformer.custom_scheduler import CustomScheduler


MAX_LENGTH = 40
D_MODEL = 256
NUM_LAYERS = 6
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1
EPOCHS = 100
BATCH_SIZE = 64
BUFFER_SIZE = 20000


def accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


def evaluate(sentence, model, tokenizer, start_token, end_token):
    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(start_token + tokenizer.encode(sentence) + end_token, axis=0)

    output = tf.expand_dims(start_token, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)
        predictions = predictions[:, -1:, :]

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, end_token[0]):
            break

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence, model, tokenizer, start_token, end_token):
    prediction = evaluate(sentence, model, tokenizer, start_token, end_token)

    predicted_sentence = tokenizer.decode([i for i in prediction if i< tokenizer.vocab_size])

    print('Input: {0}'.format(sentence))
    print('Output: {0}'.format(predicted_sentence))

    return predicted_sentence


if __name__ == '__main__':
    question_list, answer_list = get_dataset_of_korean_chatbot()

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(question_list + answer_list, target_vocab_size=2**13)

    start_token = [tokenizer.vocab_size]
    end_token = [tokenizer.vocab_size + 1]
    vocab_size = tokenizer.vocab_size + 2

    tokenized_input_list = list()
    tokenized_output_list = list()

    for input_sentence, output_sentence in zip(question_list, answer_list):
        tokenized_input = start_token + tokenizer.encode(input_sentence) + end_token
        tokenized_output = start_token + tokenizer.encode(output_sentence) + end_token

        tokenized_input_list.append(tokenized_input)
        tokenized_output_list.append(tokenized_output)

    tokenized_input_list = tf.keras.preprocessing.sequence.pad_sequences(tokenized_input_list, maxlen=MAX_LENGTH, padding='post')
    tokenized_output_list = tf.keras.preprocessing.sequence.pad_sequences(tokenized_output_list, maxlen=MAX_LENGTH, padding='post')

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': tokenized_input_list,
            'decoder_inputs': tokenized_output_list[:, :-1]
        },
        {
            'outputs': tokenized_output_list[:, 1:]
        }
    ))
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    with tf.device('/device:GPU:0'):
        tf.keras.backend.clear_session()

        model = transformer(
            vocab_size=vocab_size,
            num_layers=NUM_LAYERS,
            dff=DFF,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dropout=DROPOUT
        )

        learning_rate_scheduler = CustomScheduler(D_MODEL)
        optimizer = tf.keras.optimizers.Adam(learning_rate_scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
        model.fit(dataset, epochs=EPOCHS)

    sentence_list = list()
    sentence_list.append("영화 볼래?")
    sentence_list.append("고민이 있어.")
    sentence_list.append("너무 화가난다.")
    sentence_list.append("오늘은 카페갈까?")
    sentence_list.append("나는 오늘 딥러닝 공부를 할꺼야. 어떤 유형부터 추천해줄수 있어?")

    for sentence in sentence_list:
        output = predict(sentence, model, tokenizer, start_token, end_token)
