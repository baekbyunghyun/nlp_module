import urllib.request

import pandas as pd


if __name__ == '__main__':
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
                               filename=r'../dataset/naver_movie_review_train.txt')
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
                               filename="../dataset/naver_movie_review_test.txt")

    train_data = pd.read_table(r'../dataset/naver_movie_review_train.txt')
    test_data = pd.read_table(r'../dataset/naver_movie_review_test.txt')

    print('Number of train: {0}'.format(len(train_data)))
    print('Number of test: {0}'.format(len(test_data)))

    print(train_data[:5])
