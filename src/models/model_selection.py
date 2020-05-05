import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.utils.utils import AppendDict
from src.visualization.visualize import plot_scatter
from src.data.preprocess import preprocess, STOPWORDS

SEED = 5
MAX_TOPICS = 25
AFFINITY = 'euclidean'


def main(documents, vect_model, true_labels):
    topics_count = list(range(3, MAX_TOPICS + 1, 3))
    grid_search_res = {
        'silhouette_score': AppendDict(),
        'calinski_harabasz_score': AppendDict(),
        'davies_bouldin_score': AppendDict(),
        'adjusted_rand_score': AppendDict(),
        'fowlkes_mallows_score': AppendDict(),
        'normalized_mutual_info_score': AppendDict()
    }

    for topics in tqdm(topics_count):
        lda = LDA(n_components=topics,
                  max_iter=30,
                  n_jobs=-1,
                  learning_method='batch',
                  verbose=0,
                  random_state=SEED)

        term_doc_matrix = vect_model.transform(documents)
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
            grid_search_res['silhouette_score'][linkage] += [silhoette_]
            grid_search_res['calinski_harabasz_score'][linkage] += [calinski_]
            grid_search_res['davies_bouldin_score'][linkage] += [davies_]

            if true_labels is not None:
                rand_ = metrics.adjusted_rand_score(true_labels, labels)
                fowkels_ = metrics.fowlkes_mallows_score(true_labels, labels)
                mutual_ = metrics.normalized_mutual_info_score(true_labels, labels)
                grid_search_res['adjusted_rand_score'][linkage] += [rand_]
                grid_search_res['fowlkes_mallows_score'][linkage] += [fowkels_]
                grid_search_res['normalized_mutual_info_score'][linkage] += [mutual_]

    for metric_name in grid_search_res:
        if len(grid_search_res[metric_name]) == 0:
            continue
        print(metric_name.upper())
        plot_scatter(x=topics_count,
                     ys=list(grid_search_res[metric_name].values()),
                     labels=list(grid_search_res[metric_name].keys()),
                     title=metric_name,
                     xaxis='num_topics',
                     yaxis='score')


PATH = Path(os.getcwd())
df = pd.read_csv(PATH / 'data/raw/total.csv', sep=',')
print(df.columns)
true_labels = df['topic']
documents = df['title'] + ' ' + df['content']
if 'preproc' not in df.columns:
    documents = preprocess(documents.values, stem=True, lemm=False)
    df['preproc'] = documents
    df.to_csv(PATH / 'data/raw/total.csv', sep=',')

count_vect = CountVectorizer(input='content', stop_words=STOPWORDS)
tf_idf_vect = TfidfVectorizer(input='content', stop_words=STOPWORDS)
vect_model = count_vect
data_vectorized = vect_model.fit_transform(documents)
main(documents, vect_model, true_labels=true_labels)  # TODO try to change
