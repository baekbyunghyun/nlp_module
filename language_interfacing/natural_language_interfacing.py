import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizerFast, TFBertForSequenceClassification

from language_interfacing.download_sample_dataset import get_dataset_of_kakaobrain_nli


MAX_LENGTH_OF_SEQUENCE = 128
PRETRAINED_MODEL_NAME = r'klue/bert-base'


if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)

    X_train, y_train, X_test, y_test, X_val, y_val = get_dataset_of_kakaobrain_nli(tokenizer=tokenizer, max_length_of_sequence=MAX_LENGTH_OF_SEQUENCE)

    with tf.device('/device:GPU:0'):
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

        model = TFBertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=3, from_pt=True)
        model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])

        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.001,
            patience=2
        )

        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

    results = model.evaluate(X_test, y_test)
    print(results)

