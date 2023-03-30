import numpy as np

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."


if __name__ == '__main__':
    sentences = sent_tokenize(raw_text)

    preprocessed_sentences = list()
    stop_words = set(stopwords.words('english'))

    for sentence in sentences:
        tokenized_sentence = word_tokenize(sentence)

        result = list()
        for word in tokenized_sentence:
            word = word.lower()
            if word in stop_words:
                continue

            if len(word) <= 2:
                continue

            result.append(word)

        preprocessed_sentences.append(result)

    print(preprocessed_sentences)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(preprocessed_sentences)

    encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
    print(encoded)

    padded = pad_sequences(encoded, padding='post')
    print(padded)

    padded = pad_sequences(encoded, padding='post', truncating='post', maxlen=5)
    print(padded)