import os
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def add_row(df, current_lines):
    new_article = pd.Series(
        {'topic': current_lines[1],
         'cat': current_lines[2],
         'title': current_lines[3],
         'url': current_lines[4],
         'content': '. '.join(current_lines[5:])}
    )
    df = df.append(new_article, ignore_index=True)
    return df


NUMBER_RE = r'[0-9]+'
PATH = Path(os.getcwd())
file = open(PATH / 'data/raw/contents_wrong_format.csv')
i = 0
for i in range(5):  # skip first 5 rows (headers)
    file.readline()

df = pd.DataFrame(columns=['topic', 'cat', 'title', 'url', 'content'])

topics = ['астрономия', 'биология', 'химия', 'физика', 'математика', 'география', 'тарих',
          'әдәбият', 'фәлсәфә', 'психология', 'сәясәт', 'икътисад', 'хокук']
ind = -1
cur_lines = list()

all_lines = file.readlines()
for i in tqdm(range(1, len(all_lines))):
    line = all_lines[i].strip()
    prev_line = all_lines[i - 1].strip()

    if line in topics and bool(re.fullmatch(NUMBER_RE, prev_line)) and len(cur_lines) >= 3:
        df = add_row(df, cur_lines)
        cur_lines = [prev_line, line]
        ind += 1
    else:
        cur_lines.append(line)


df = add_row(df, cur_lines)

df.to_csv(PATH / 'data/raw/contents.csv')
print(df.head(20))
