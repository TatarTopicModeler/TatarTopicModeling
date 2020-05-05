import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from pathlib import Path
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.visualization.visualize import plot_scatter
from src.data.preprocess import preprocess, STOPWORDS

SEED = 5
TOPICS = 6


def main(df, use_true=False):
    documents = df['preproc']
    names = df['title']
    true_labels_map = dict([(x, i) for i, x in enumerate(set(df['topic']))])
    true_labels = [true_labels_map[x] for x in df['topic']]

    lda = LDA(n_components=TOPICS,
              max_iter=30,
              n_jobs=6,
              learning_method='batch',
              # verbose=1,
              random_state=SEED,
              learning_decay=0.7)
    tSNE = TSNE(n_components=2,
                # perplexity=50,
                n_jobs=-1,
                random_state=SEED)

    term_doc_matrix = vect_model.transform(documents)
    embeddings = lda.fit_transform(term_doc_matrix)

    for metric in ["cosine"]:  # + ["euclidean", "manhattan"]:
        print('=' * 20)
        print(metric)
        X = StandardScaler().fit_transform(embeddings)
        ac = AgglomerativeClustering(n_clusters=TOPICS,
                                     linkage='average',
                                     affinity=metric
                                     )
        ac = ac.fit(X)

        lda_labels = np.argmax(embeddings, axis=1)
        clust_labels = ac.labels_

        labels_variants = [
            ('LDA labels', lda_labels, X),
            ('CLUSTERING labels', clust_labels, X)
        ]
        for name, labels, X in labels_variants:
            print(name)
            print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels, metric=metric), '\n')
            X = tSNE.fit_transform(embeddings)

            draw_df = pd.DataFrame()
            draw_df['x'] = X[:, 0]
            draw_df['y'] = X[:, 1]
            if use_true:
                draw_df['label'] = true_labels
            else:
                draw_df['label'] = labels
            draw_df['Name'] = names

            # fig = px.scatter(draw_df, x='x', y='y', color='label', hover_data=['Name'], title=f'{name}. TOPICS: {TOPICS}')

            fig = go.Figure(data=go.Scatter(
                x=draw_df['x'],
                y=draw_df['y'],
                mode='markers',
                marker_color=draw_df['label'],
                text=draw_df['Name'],
                marker={'size': 10,
                        'symbol': 'circle-dot'},
            ))

            """
            unique_labels = set(labels)
            fig = go.Figure()
            for k in unique_labels:
                class_member_mask = (labels == k)
                xy = X[class_member_mask] 
                _ = fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1],
                                             mode='markers',
                                             name=str(k),
                                             marker={'size': 12, 
                                                     'symbol': 'circle-dot'},
                                             ))
            """

            _ = fig.update_layout(title=f'{name}. TOPICS: {TOPICS}. True labels: {use_true}',
                                  xaxis_title='x',
                                  yaxis_title='y',
                                  width=1000,
                                  height=1000,
                                  coloraxis={'colorscale': 'viridis'})
            fig.show()


PATH = Path(os.getcwd())
df = pd.read_csv(PATH / 'data/raw/total.csv', sep=',')
print(df.columns)
true_labels = df['topic']
documents = df['title'] + ' ' + df['content']
if 'preproc' not in df.columns:
    documents = preprocess(documents.values, stem=True, lemm=False)
    df['preproc'] = documents
    df.to_csv(PATH / 'data/raw/total.csv', sep=',')

df.dropna(inplace=True)
documents = df['preproc']

count_vect = CountVectorizer(input='content', stop_words=STOPWORDS)
tf_idf_vect = TfidfVectorizer(input='content', stop_words=STOPWORDS)
vect_model = count_vect
data_vectorized = vect_model.fit_transform(documents)

main(df, use_true=True)
