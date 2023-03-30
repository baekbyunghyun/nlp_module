import numpy as np

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.sparse import coo_matrix
from sklearn.datasets import load_svmlight_file
from tensorflow.python.client import device_lib


TRAIN_FILE_PATH = r'dataset/train_naver_movie_review.spa'
TEST_FILE_PATH = r'dataset/test_naver_movie_review.spa'

embedding_dim = 100
hidden_units = 128
vocab_size = 19416

train_X, train_y = load_svmlight_file(TRAIN_FILE_PATH)
train_X = np.array(coo_matrix(train_X, dtype=np.float32).todense())

test_X, test_y = load_svmlight_file(TEST_FILE_PATH)
test_X = np.array(coo_matrix(test_X, dtype=np.float32).todense())

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(train_X, train_y, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(test_X, test_y)[1]))