from collections import Counter

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer


raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."


def use_dictionary(sentences):
    vocab = dict()
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
            if word not in vocab:
                vocab[word] = 0

            vocab[word] += 1

        preprocessed_sentences.append(result)

    print(preprocessed_sentences)
    print(vocab)

    sorted_vocab = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
    print(sorted_vocab)

    word_to_index = dict()
    i = 0
    for (word, frequency) in sorted_vocab:
        if frequency <= 1:
            continue

        i = i + 1
        word_to_index[word] = i

    print(word_to_index)

    vocab_size = 5
    words_frequency = [word for word, index in word_to_index.items() if index >= vocab_size + 1]

    for w in words_frequency:
        del word_to_index[w]

    print(word_to_index)

    word_to_index['OOV'] = len(word_to_index) + 1
    print(word_to_index)

    encoded_sentences = list()
    for sentence in preprocessed_sentences:
        encoded_sentence = list()
        for word in sentence:
            try:
                encoded_sentence.append(word_to_index[word])

            except KeyError:
                encoded_sentence.append(word_to_index['OOV'])

        encoded_sentences.append(encoded_sentence)

    print(encoded_sentences)


def use_counter(sentences):
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

    all_word_list = sum(preprocessed_sentences, [])
    print(all_word_list)

    vocab = Counter(all_word_list)
    print(vocab)

    vocab = vocab.most_common(5)
    print(vocab)

    word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
    print(word_to_index)


def use_keras(sentences):
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

    tokenizer = Tokenizer(num_words=5+1, oov_token='OOV')
    tokenizer.fit_on_texts(preprocessed_sentences)

    print(tokenizer.word_index)
    print(tokenizer.word_counts)

    print(tokenizer.texts_to_sequences(preprocessed_sentences))


if __name__ == '__main__':
    sentences = sent_tokenize(raw_text)
    print(sentences)

    use_keras(sentences)
