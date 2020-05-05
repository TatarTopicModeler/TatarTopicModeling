from nltk.stem import SnowballStemmer, WordNetLemmatizer
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import unicodedata
import nltk
import os
import re

PATH = Path(os.getcwd())  # Project path
tt_stopwords_file = open(PATH / 'data/external/stopwords_tt.txt', 'r')  # Stopwords file
ru_stopwords_file = open(PATH / 'data/external/stopwords_ru.txt', 'r')  # Stopwords file (all nltk words are included)
dataset_stopwords_file = open(PATH / 'data/interim/stopwords_dataset.txt', 'r')  # Dataset specific stopwords

tt_stopwords = [line.strip() for line in tt_stopwords_file]
ru_stopwords = [line.strip() for line in ru_stopwords_file]
dataset_stopwords = [line.strip() for line in dataset_stopwords_file]
STOPWORDS = tt_stopwords + ru_stopwords + dataset_stopwords

URL_EXP = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
russian_stemmer = SnowballStemmer('russian')
english_stemmer = SnowballStemmer('english')

ENGLISH_ALPHA = {chr(x) for x in range(ord('a'), ord('z') + 1)}
RUSSIAN_ALPHA = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
TATAR_ALPHA = set('аәбвгдеёжҗзийклмнңоөпрстуүфхһцчшщъыьэюя')
ALPHA = ENGLISH_ALPHA.union(RUSSIAN_ALPHA).union(TATAR_ALPHA)  # union of three alphabets
PUNCT = r'''!"#$%&\'()*+,-./:;<=>?@\\][^_`{|}~«»—“”•'''  # punctuation marks


def remove_accents(text):
    nfkd_form = unicodedata.normalize('NFKD', text)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def normalize(text):
    text = text.lower()
    text = re.sub(URL_EXP, ' ', text)  # remove URLS
    text = re.sub(f'[{PUNCT}]', ' ', text)  # remove all punctuation signs
    text = re.sub('[0-9]', ' ', text)  # replace all digits with spaces
    text = remove_accents(text)
    result = " ".join([x.lower() for x in text.split()])  # lower all letters and delete all doubled spaces
    # TODO delete stress marks
    return result


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


if __name__ == '__main__':
    t1 = """Савытка йомырка сыталар, сөт салалар, май, шикәр комы, тоз, азрак чүпрә өстиләр (күп вакытта чүпрә салмыйлар да). Шул катнашманы шикәре, тозы һәм чүпрәсе эреп беткәнче яхшылап болгаталар. Аннары он кушып, токмач камырыннан йомшаграк итеп камыр басалар. Камырны 100—150 г лы кисәкләргә бүләләр дә бармак калынлыгында баусыман тәгәрәтәләр һәм урман чикләвеге кадәр итеп турыйлар. Кайнап торган майга салалар, болгата-болгата, кызарганчы пешерәләр.
    Әзер бавырсакны тишекле зур чүмечтә сөзәләр, мае агып беткәч, өстенә шикәр оны сибергә була. Бавырсакны чәй янына чыгаралар. Аны катык һәм сөт белән дә ашарга була. Ул юлга алу өчен дә әйбәт.
    1 кг югары сортлы онга: 10 йомырка, 130—140 г сөт, 30—35 г шикәр комы, 30 г сары май, 5 г чүпрә, тол, пешерү өчен 180—200 г май."""
    t2 = pd.read_csv(PATH / 'data/raw/contents.csv', sep=',')['content'].iloc[0]
    prep = preprocess(t2, stem=True, lemm=False)
    print(prep)
    print(remove_accents('кәбестә́'))
