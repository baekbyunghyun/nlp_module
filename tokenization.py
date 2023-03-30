from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag

from konlpy.tag import Okt
from konlpy.tag import Kkma

# from tensorflow.keras.preprocessing.text import text_to_word_sequence

import kss


# corpus = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
#
#
# print('단어 토큰화1 :', word_tokenize(corpus))
# print('단어 토큰화2 :', WordPunctTokenizer().tokenize(corpus))
# print('단어 토큰화3 :', text_to_word_sequence(corpus))

# tokenizer = TreebankWordTokenizer()
#
# text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
# print('트리뱅크 워드토크나이저 :',tokenizer.tokenize(text))

# text = "딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?"
#
# print('문장 토큰화1 : ', kss.split_sentences(text))

# text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
#
# tokenized_sentence = word_tokenize(text)
# print('단어 토큰화: ', word_tokenize(text))
# print('품사 태깅: ', pos_tag(tokenized_sentence))

okt = Okt()
kkma = Kkma()

text = "열심히 코딩한 당신, 연휴에는 여행을 가봐요"

print('okt 형태소 분석: ', okt.morphs(text))
print('okt 품사 태깅: ', okt.pos(text))
print('okt 명사 추출: ', okt.nouns(text))