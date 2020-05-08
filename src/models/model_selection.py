import enum
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.visualization.visualize import plot_scatter
from src.data.preprocess import preprocess, STOPWORDS

SEED = 5
MAX_TOPICS = 35
AFFINITY = 'euclidean'


class Scores(enum.Enum):
    silhouette = enum.auto()
    calinski_harabasz = enum.auto()
    davies_bouldin = enum.auto()
    adjusted_rand = enum.auto()
    fowlkes_mallows = enum.auto()
    normalized_mutual_info = enum.auto()


def main(documents, vect_model, true_labels):
    topics_count = list(range(3, MAX_TOPICS + 1, 3))
    grid_search_res = {score: defaultdict(list) for score in Scores}
    term_doc_matrix = vect_model.fit_transform(documents)

    for topics in tqdm(topics_count, desc='topics_count'):
        lda = LDA(n_components=topics,
                  max_iter=30,
                  n_jobs=-1,
                  learning_method='batch',
                  verbose=0,
                  random_state=SEED)

        embeddings = lda.fit_transform(term_doc_matrix)
        X = StandardScaler().fit_transform(embeddings)

        for linkage in ['ward', 'complete', 'average', 'LDA argmax']:
            if linkage == 'LDA argmax':
                labels = np.argmax(embeddings, axis=1)
            else:
                ac = AgglomerativeClustering(
                    n_clusters=topics,
                    linkage=linkage,
                    affinity=AFFINITY
                )
                ac = ac.fit(X)
                labels = ac.labels_

            silhoette_ = metrics.silhouette_score(X, labels, metric=AFFINITY)
            calinski_ = metrics.calinski_harabasz_score(X, labels)
            davies_ = metrics.davies_bouldin_score(X, labels)

            grid_search_res[Scores.silhouette][linkage] += [silhoette_]
            grid_search_res[Scores.calinski_harabasz][linkage] += [calinski_]
            grid_search_res[Scores.davies_bouldin][linkage] += [davies_]

            if true_labels is not None:
                rand_ = metrics.adjusted_rand_score(true_labels, labels)
                fowkels_ = metrics.fowlkes_mallows_score(true_labels, labels)
                mutual_ = metrics.normalized_mutual_info_score(true_labels, labels)

                grid_search_res[Scores.adjusted_rand][linkage] += [rand_]
                grid_search_res[Scores.fowlkes_mallows][linkage] += [fowkels_]
                grid_search_res[Scores.normalized_mutual_info][linkage] += [mutual_]

    for metric_name in grid_search_res:
        if len(grid_search_res[metric_name]) == 0:
            continue
        print(metric_name.name.upper())
        plot_scatter(x=topics_count,
                     ys=list(grid_search_res[metric_name].values()),
                     labels=list(grid_search_res[metric_name].keys()),
                     title=metric_name.name,
                     xaxis='num_topics',
                     yaxis='score')


PATH_TOTAL_RAW = Path('data/processed/total.csv')
PATH_TOTAL_PROCESSED = Path('data/processed/total.csv')

PATH = Path(os.getcwd())
filename = 'contents.csv'
if filename in os.listdir(PATH / 'data/processed'):
    df = pd.read_csv(PATH / 'data/processed' / filename)
else:
    df = pd.read_csv(PATH / 'data/raw' / filename)
    df['preproc'] = preprocess(df['content'].values, lemm=True)
    df.to_csv(PATH / 'data/processed' / filename)
df.dropna(inplace=True)

df = df[df['title'].str.len() > 1]  # remove articles about letters
documents = df['preproc'].values  # .astype('U')
print(type(documents), type(STOPWORDS))
count_vect = CountVectorizer(input='content')  # , stop_words=STOPWORDS)
tf_idf_vect = TfidfVectorizer(input='content')  # , stop_words=STOPWORDS)
vect_model = count_vect

main(list(documents), vect_model, true_labels=df['topic'])
