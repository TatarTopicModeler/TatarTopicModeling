from functools import reduce
import pandas as pd

import requests
from bs4 import BeautifulSoup

BASE_URL = 'https://tt.wikipedia.org/'


def filter_url(url: str):
    _stop = ['%D0%A2%D3%A9%D1%80%D0%BA%D0%B5%D0%BC']
    cond = [s not in url for s in _stop]
    cond = reduce(lambda x, y: x * y, cond)
    return bool(cond)


def process_cat(page_url):
    cont = requests.get(page_url).content
    soup = BeautifulSoup(cont)

    articles_block = soup.find('div', dir='ltr', class_='mw-content-ltr')
    links = articles_block.find_all('a', class_=None)
    links = [(l.get('title'), l.get('href')) for l in links]

    links = list(set(links))
    links = [l for l in links if filter_url(l[1])]

    df = pd.DataFrame(columns=['title', 'url'])
    df['title'] = [l[0] for l in links]
    df['url'] = [BASE_URL + l[1] for l in links]

    print(df)



if __name__ == '__main__':
    cats = [
        'https://tt.wikipedia.org/wiki/%D0%A2%D3%A9%D1%80%D0%BA%D0%B5%D0%BC:%D2%BA%D3%A9%D0%BD%D3%99%D1%80%D0%BB%D3%99%D1%80'
    ]
    process_cat(cats[0])
