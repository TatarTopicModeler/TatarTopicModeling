import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from tqdm import tqdm
from pathlib import Path
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from collections import OrderedDict, Counter
from src.models.topic_modeler import TopicModeler
from src.visualization.visualize import plot_hist
from src.data.preprocess import preprocess, STOPWORDS

SEED = 5
TOPICS = 13
PERPLEXITY = 35
COLORMAP = px.colors.qualitative.Light24
COLORMAP = [COLORMAP[i] for i in
            (0, 1, 2, 3, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 21, 22)]


def plot(documents, vect_model, names, true_labels, use_true=False):
    true_label_to_id = OrderedDict([(x, i) for i, x in enumerate(sorted(set(true_labels)))])
    true_id_to_label = OrderedDict([(i, x) for i, x in enumerate(sorted(set(true_labels)))])
    true_labels = np.array([true_label_to_id[x] for x in true_labels])

    lda = LDA(n_components=TOPICS,
              max_iter=30,
              n_jobs=6,
              learning_method='batch',
              verbose=0,
              random_state=SEED,
              learning_decay=0.7)

    data_vectorized = vect_model.fit_transform(list(documents))
    embeddings = lda.fit_transform(data_vectorized)

    for metric in tqdm(['cosine']):  # , 'euclidean', 'manhattan']):
        X = StandardScaler().fit_transform(embeddings)

        tSNE = TSNE(n_components=2,
                    metric=metric,
                    perplexity=PERPLEXITY,
                    n_jobs=-1,
                    init='random',
                    random_state=SEED)
        ac = AgglomerativeClustering(n_clusters=TOPICS,
                                     linkage='average',
                                     affinity=metric
                                     )
        ac = ac.fit(X)

        lda_labels = np.argmax(embeddings, axis=1)
        clust_labels = ac.labels_

        labels_variants = [
            ('LDA labels', lda_labels, X),
            # ('CLUSTERING labels', clust_labels, X)
        ]
        for name, labels, X in labels_variants:
            print('=' * 20)
            print(name, metric)
            print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels, metric=metric), '\n')
            X = tSNE.fit_transform(X)

            if use_true:
                unique_labels = set(true_labels)
                labels = true_labels
                id_to_label = true_id_to_label
            else:
                unique_labels = set(labels)
                id_to_label = dict([(i, i) for i in range(len(unique_labels))])

            fig = go.Figure()
            for k in unique_labels:
                class_member_mask = (labels == k)
                xy = X[class_member_mask]
                _ = fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1],
                                             mode='markers',
                                             name=id_to_label[k],
                                             marker={'size': 12,
                                                     'symbol': 'circle',
                                                     'opacity': 0.8,
                                                     'line': {'width': 1,
                                                              'color': 'SlateGrey'},
                                                     'color': COLORMAP[(len(COLORMAP) + k) % len(COLORMAP)]
                                                     },
                                             hovertext=np.array(names)[class_member_mask]
                                             ))
            title = f'{name}. TOPICS: {TOPICS}. Metric: {metric}. Perplexity: {PERPLEXITY}. True labels: {use_true}'
            _ = fig.update_layout(title=title,
                                  xaxis_title='x',
                                  yaxis_title='y',
                                  width=1000,
                                  height=1000,
                                  showlegend=True,
                                  coloraxis={'colorscale': 'viridis'})
            fig.show()
            fig.write_image('images/best_clustering.png')


def topics_distribution(documents, vect_model):
    lda = LDA(n_components=TOPICS,
              max_iter=30,
              n_jobs=6,
              learning_method='batch',
              # verbose=1,
              random_state=SEED,
              learning_decay=0.7)

    data_vectorized = vect_model.fit_transform(documents)
    lda.fit(data_vectorized)
    topic_modeler = TopicModeler(vect_model, lda)
    prob = [topic_modeler(doc) for doc in documents]
    most_prob = [np.argmax(x) for x in prob]
    plot_hist(most_prob, 'Topics distribution', size=(700, 500), file='best_topics_distribution')
    most_prob_cnt = Counter(most_prob)
    print(most_prob_cnt)


PATH = Path(os.getcwd())
filename = 'contents.csv'
if filename in os.listdir(PATH / 'data/processed'):
    df = pd.read_csv(PATH / 'data/processed' / filename)
else:
    df = pd.read_csv(PATH / 'data/raw' / filename)
    df['preproc'] = preprocess(df['content'].values, stem=True)
    df.to_csv(PATH / 'data/processed' / filename)
df.dropna(inplace=True)

df = df[~df['title'].str.contains('Калып:')]  # remove templates
documents = df['preproc'].values

count_vect = CountVectorizer(input='content')  # , stop_words=STOPWORDS)
tf_idf_vect = TfidfVectorizer(input='content')  # , stop_words=STOPWORDS)
vect_model = count_vect

plot(documents, vect_model, df['title'], df['topic'], use_true=True)
topics_distribution(documents, vect_model)
