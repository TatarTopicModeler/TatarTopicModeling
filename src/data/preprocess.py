from nltk.stem import SnowballStemmer, WordNetLemmatizer
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import unicodedata
import nltk
import os
import re

PATH = Path(os.getcwd())
PATH_STOPWORDS_TT = Path('data/external/stopwords_tt.csv') # Stopwords file
PATH_STOPWORDS_RU = Path('data/external/stopwords_ru.csv') # Stopwords file (all nltk words are included)
PATH_STOPWORDS_INTERIM = Path('data/interim/stopwords_dataset.csv')

dataset_stopwords_file = open(PATH_STOPWORDS_INTERIM, 'r')  # Dataset specific stopwords

tt_stopwords = pd.read_csv(PATH_STOPWORDS_TT, header=None)
ru_stopwords = pd.read_csv(PATH_STOPWORDS_RU, header=None)
# interim_stopwords = pd.read_csv(PATH_STOPWORDS_INTERIM, header=None)


STOPWORDS = pd.DataFrame().append([tt_stopwords, ru_stopwords])

URL_EXP = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
russian_stemmer = SnowballStemmer('russian')
english_stemmer = SnowballStemmer('english')

ENGLISH_ALPHA = {chr(x) for x in range(ord('a'), ord('z') + 1)}
RUSSIAN_ALPHA = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
TATAR_ALPHA = set('аәбвгдеёжҗзийклмнңоөпрстуүфхһцчшщъыьэюя')
ALPHA = ENGLISH_ALPHA.union(RUSSIAN_ALPHA).union(TATAR_ALPHA)  # union of three alphabets
PUNCT = r'''!"#$%&\'()*+,-./:;<=>?@\\][^_`{|}~«»—“”•'''  # punctuation marks


def remove_stress(text):
    nfkd_form = unicodedata.normalize('NFKD', text)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def normalize(text):
    text = text.lower()
    text = re.sub(URL_EXP, ' ', text)  # remove URLS
    text = re.sub(f'[{PUNCT}]', ' ', text)  # remove all punctuation signs
    text = re.sub('[0-9]', ' ', text)  # replace all digits with spaces
    text = re.sub(r'\s+', ' ', text)
    text = remove_stress(text)
    return text

def is_english(word):
    return sum([int(char in ENGLISH_ALPHA) for char in word]) >= len(word) // 2


def stemming(words):  # TODO adapt for tatar (implement a new one or find edit distance with dictionary word
    return [
        english_stemmer.stem(word) if is_english(word) else russian_stemmer.stem(word)
        for word in words
    ]


def lemmatize(text):  # TODO не работает
    return WordNetLemmatizer().lemmatize(text, pos='v')


def remove_stopwords(words):
    return [word for word in words if word not in STOPWORDS]


def preprocess_document(text, stem=False, lemm=False):
    text = normalize(text)
    words = nltk.word_tokenize(text)
    words = remove_stopwords(words)
    if stem:
        words = stemming(words)
    if lemm:
        words = [lemmatize(word) for word in text]
    words = [word for word in words if len(word) > 1]
    return ' '.join(words)


def preprocess(documents, stem=False, lemm=False):
    return [preprocess_document(str(doc), stem, lemm) for doc in tqdm(documents)]


df = pd.read_csv(PATH / 'data/raw/total.csv')
df['preproc'] = preprocess(df['content'], stem=False, lemm=False)
words = [x for d in df['preproc'] for x in nltk.word_tokenize(d)]
cnt = Counter(words)
df.to_csv(PATH / 'data/processed/total.csv')

for x in dict(cnt.most_common(100)).items():
    print(x)
