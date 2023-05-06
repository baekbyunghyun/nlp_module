import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizerFast, TFBertForSequenceClassification, TextClassificationPipeline


TRAIN_DATASET_FILE_PATH = r'../dataset/naver_movie_review_train.txt'
TEST_DATASET_FILE_PATH = r'../dataset/naver_movie_review_test.txt'
MAX_LENGTH_OF_SEQUENCE = 128
PRETRAINED_MODEL_NAME = r'klue/bert-base'
SAVED_MODEL_NAME = r'nsmc_model/bert-base'


def get_train_test_of_naver_movie_review_dataset(train_file_path=None, test_file_path=None):
    if (train_file_path is None) or (test_file_path is None):
        raise Exception('Invalid dataset file path.')

    train_df = pd.read_table(train_file_path)
    test_df = pd.read_table(test_file_path)

    train_df.drop_duplicates(subset=['document'], inplace=True)
    train_df = train_df.dropna(how='any')

    test_df = test_df.dropna(how='any')

    return train_df['document'].tolist(), train_df['label'].tolist(), test_df['document'].tolist(), test_df['label'].tolist()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_train_test_of_naver_movie_review_dataset(train_file_path=TRAIN_DATASET_FILE_PATH, test_file_path=TEST_DATASET_FILE_PATH)

    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)

    X_train = tokenizer(X_train, truncation=True, padding=True)
    X_test = tokenizer(X_test, truncation=True, padding=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(X_train),
        y_train
    ))

    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(X_test),
        y_test
    ))

    with tf.device('/device:GPU:0'):
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

        model = TFBertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME)
        model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])

        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.001,
            patience=2
        )

        model.fit(train_dataset.shuffle(10000).batch(32),
                  validation_data=val_dataset.shuffle(10000).batch(32),
                  epochs=2, batch_size=32, callbacks=[early_stopping])

    results = model.evaluate(val_dataset.batch(1024))
    print(results)

    model.save_pretrained(SAVED_MODEL_NAME)
    tokenizer.save_pretrained(SAVED_MODEL_NAME)

    loaded_tokenizer = BertTokenizerFast.from_pretrained(SAVED_MODEL_NAME)
    loaded_model = TFBertForSequenceClassification.from_pretrained(SAVED_MODEL_NAME)

    text_classifier = TextClassificationPipeline(
        tokenizer=loaded_tokenizer,
        model=loaded_model,
        framework='tf',
        return_all_scores=True
    )

    print(text_classifier('평점이 이게 뭐야?')[0])

