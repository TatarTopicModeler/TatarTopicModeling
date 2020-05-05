from typing import Tuple
import plotly.graph_objs as go
from plotly.offline import iplot


def layout(title_raw, width, height):
    return go.Layout(title=title_raw, autosize=False,
                     width=width, height=height,
                     margin=dict(l=65, r=50, b=65, t=90))


def plot_hist(data, title, histnorm=None, size: Tuple = None):
    def hist(x):
        return [go.Histogram(x=x, histnorm=histnorm)]

    if size is not None:
        width, height = size
    else:
        width, height = (600, 600)
    fig = go.Figure(data=hist(data), layout=layout(title, width, height))
    iplot(fig)


def plot_bar(x, y, title, size: Tuple = None):
    def bar(x, y):
        return [go.Bar(x=x, y=y)]

    if size is not None:
        width, height = size
    else:
        width, height = (600, 600)
    fig = go.Figure(data=bar(x, y), layout=layout(title, width, height))
    iplot(fig)


def plot_scatter(x, ys, labels, title, xaxis='', yaxis=''):
    fig = go.Figure()
    _ = fig.update_layout(title=title,
                          xaxis_title=xaxis,
                          yaxis_title=yaxis)
    for i in range(len(ys)):
        _ = fig.add_trace(go.Scatter(x=x, y=ys[i],
                                     mode='lines+markers',
                                     name=labels[i]))
    fig.show()
