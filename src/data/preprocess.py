from nltk.stem import SnowballStemmer, WordNetLemmatizer
from src.data.tatar_stemming import tatar_stemmer
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import unicodedata
import nltk
import os
import re

PATH = Path(os.getcwd())
PATH_STOPWORDS_TT = Path('data/external/stopwords_tt.csv')  # Stopwords file
PATH_STOPWORDS_RU = Path('data/external/stopwords_ru.csv')  # Stopwords file (all nltk words are included)
PATH_STOPWORDS_EN = Path('data/external/stopwords_en.csv')  # Stopwords file (all nltk words are included)
PATH_STOPWORDS_INTERIM = Path('data/interim/stopwords_dataset.csv')

dataset_stopwords_file = open(PATH_STOPWORDS_INTERIM, 'r')  # Dataset specific stopwords

tt_stopwords = pd.read_csv(PATH_STOPWORDS_TT, header=None)
ru_stopwords = pd.read_csv(PATH_STOPWORDS_RU, header=None)
en_stopwords = pd.read_csv(PATH_STOPWORDS_EN, header=None)
interim_stopwords = pd.read_csv(PATH_STOPWORDS_INTERIM, header=None)

STOPWORDS = pd.DataFrame().append([tt_stopwords, ru_stopwords, en_stopwords, interim_stopwords])
STOPWORDS = STOPWORDS.values
# STOPWORDS = tuple(STOPWORDS.values)

URL_EXP = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
russian_stemmer = SnowballStemmer('russian')
english_stemmer = SnowballStemmer('english')

EN_ALPHA_REGEX = r'a-z'
RU_ALPHA_REGEX = r'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
TT_ALPHA_REGEX = r'аәбвгдеёжҗзийклмнңоөпрстуүфхһцчшщъыьэюя'
TT_LATIN_ALPHA_REGEX = r'abcdefghijklmnopqrstuvwxyzäççñöüüğışş\''
ALPHA_REGEX = EN_ALPHA_REGEX + TT_ALPHA_REGEX + RU_ALPHA_REGEX + EN_ALPHA_REGEX

PUNCT = r'''!"#$%&\'()*+,-./:;<=>?@\\][^_`{|}~«»—“”•'''  # punctuation marks

TRANSLIT_RULES = {
    'ts': 'ц',
    'yü': 'ю',
    'yu': 'ю',
    'ya': 'я',
    'yä': 'я',
    'eu': 'ев',
    'ye': 'е',
    'yı': 'е',
    'şç': 'щ',
    'a': 'а',
    'b': 'б',
    'ç': 'ч',
    'c': 'җ',
    'd': 'д',
    'e': 'е',
    'ä': 'ә',
    'f': 'ф',
    'g': 'г',
    'ğ': 'г',
    'h': 'һ',
    'i': 'и',
    'y': 'й',
    'k': 'к',
    'l': 'л',
    'm': 'м',
    'n': 'н',
    'ñ': 'ң',
    'o': 'о',
    'ö': 'ө',
    'p': 'п',
    'q': 'к',
    'r': 'р',
    's': 'с',
    'ş': 'ш',
    't': 'т',
    'u': 'у',
    'v': 'в',
    'w': 'в',
    'x': 'х',
    'ü': 'ү',
    'z': 'з',
    'j': 'ж',
    'ı': 'ы',
    '\'': '',  # (э, ъ, ь)
}
wnl = WordNetLemmatizer()


def remove_stress(text, ignore='йё'):
    normalized = unicodedata.normalize('NFKC', text)
    normalized = [n for n in normalized if not unicodedata.combining(n)]
    return u''.join(normalized)


def normalize(text):
    text = re.sub(URL_EXP, ' ', text)  # remove URLS
    text = remove_stress(text)  # normalize all chars expect ['й','ё']
    text = re.sub(f'[^{ALPHA_REGEX}]', ' ', text)  # leave only letters
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text


def is_english(word):
    return sum([int(char in EN_ALPHA_REGEX) for char in word]) >= len(word) // 2


def is_latin_tatar_text(text):
    words = nltk.word_tokenize(text)
    total = sum([len(word) for word in words])
    latin = sum([
        sum([(char in TT_LATIN_ALPHA_REGEX) for char in word])
        for word in words
    ])
    return latin / total >= 0.6


def stemming(words):
    return [
        english_stemmer.stem(word) if is_english(word) else tatar_stemmer(word)
        for word in words
    ]


def translit(text):
    for from_, to_ in TRANSLIT_RULES.items():
        text = re.sub(f'{from_}', f'{to_}', text)

    return text


def remove_stopwords(words):
    return [word for word in words if word not in STOPWORDS]


def preprocess_document(text, stem=False):
    text = text.lower()
    if is_latin_tatar_text(text):
        text = translit(text)

    text = normalize(text)
    words = nltk.word_tokenize(text)
    words = remove_stopwords(words)
    if stem:
        words = stemming(words)
    words = remove_stopwords(words)
    words = [word for word in words if len(word) > 1]
    return ' '.join(words)


def preprocess(documents, stem=False):
    return [preprocess_document(str(doc), stem) for doc in tqdm(documents, desc='Preprocessing')]


if __name__ == '__main__':
    df = pd.read_csv(PATH / 'data/raw/contents.csv')
    df['preproc'] = preprocess(df['content'], stem=True)

    words = [x for d in df['preproc'] for x in nltk.word_tokenize(d)]
    cnt = Counter(words)
    df.to_csv(PATH / 'data/processed/contents.csv')

    for x in dict(cnt.most_common(100)).items():
        print(x)
