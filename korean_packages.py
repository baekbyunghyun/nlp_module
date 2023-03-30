import urllib.request

from konlpy.tag import Okt
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.normalizer import *
from ckonlpy.tag import Twitter


tokenizer = Okt()
print(tokenizer.morphs('에이비식스 이대휘 1월 최애돌 기부 요정'))

urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")

corpus = DoublespaceLineCorpus("2016-10-20.txt")

word_extractor = WordExtractor()
word_extractor.train(corpus)
word_score_table = word_extractor.extract()

print(word_score_table)

print(word_score_table["반포한"].cohesion_forward)
print(word_score_table["반포한강"].cohesion_forward)
print(word_score_table["반포한강공"].cohesion_forward)
print(word_score_table["반포한강공원"].cohesion_forward)
print(word_score_table["반포한강공원에"].cohesion_forward)

print("--")

print(word_score_table["디스"].right_branching_entropy)
print(word_score_table["디스플"].right_branching_entropy)
print(word_score_table["디스플레"].right_branching_entropy)
print(word_score_table["디스플레이"].right_branching_entropy)

print("--")

scores = {word:score.cohesion_forward for word, score in word_score_table.items()}
l_tokenizer = LTokenizer(scores=scores)
print(l_tokenizer.tokenize("국제사회와 우리의 노력들로 범죄를 척결하자", flatten=False))

maxscore_tokenizer = MaxScoreTokenizer(scores=scores)
print(maxscore_tokenizer.tokenize("국제사회와우리의노력들로범죄를척결하자"))

print(emoticon_normalize('앜ㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠㅠㅠ', num_repeats=2))

print(repeat_normalize('와하하하하하하하하하핫', num_repeats=2))
print(repeat_normalize('와하하하하하하핫', num_repeats=2))
print(repeat_normalize('와하하하하핫', num_repeats=2))

twitter = Twitter()
print(twitter.morphs('은경이는 사무실로 갔습니다.'))

twitter.add_dictionary('은경이', 'Noun')

print(twitter.morphs('은경이는 사무실로 갔습니다.'))