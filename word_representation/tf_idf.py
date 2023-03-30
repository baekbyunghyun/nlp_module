import pandas as pd # 데이터프레임 사용을 위해
from math import log # IDF 계산을 위해


def tf(t, d):
  return d.count(t)

def idf(t):
  df = 0
  for doc in docs:
    df += t in doc
  return log(N/(df+1))

def tfidf(t, d):
  return tf(t,d)* idf(t)