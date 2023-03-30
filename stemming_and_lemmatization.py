from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer


lemmatizer = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print('표제어 추출 후 :', [lemmatizer.lemmatize(word) for word in words])

print(lemmatizer.lemmatize('dies', 'v'))
print(lemmatizer.lemmatize('watched', 'v'))
print(lemmatizer.lemmatize('has', 'v'))

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

print('포터 스테머의 어간 추출 후:', [porter_stemmer.stem(w) for w in words])
print('랭커스터 스테머의 어간 추출 후:', [lancaster_stemmer.stem(w) for w in words])
