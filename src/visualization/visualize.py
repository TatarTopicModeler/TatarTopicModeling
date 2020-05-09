from typing import Tuple
import plotly.graph_objs as go
from plotly.offline import iplot


def layout(title_raw, width, height):
    return go.Layout(title=title_raw, autosize=False,
                     width=width, height=height,
                     margin=dict(l=65, r=50, b=65, t=90))


def plot_hist(data, title, histnorm=None, size: Tuple = None, file=None):
    def hist(x):
        return [go.Histogram(x=x, histnorm=histnorm)]

    if size is not None:
        width, height = size
    else:
        width, height = (600, 600)
    fig = go.Figure(data=hist(data), layout=layout(title, width, height))
    iplot(fig)
    if file is not None:
        fig.write_image(f'reports/figures/{file}.png')


def plot_bar(x, y, title, size: Tuple=None, file=None):
    def bar(x, y):
        return [go.Bar(x=x, y=y)]

    if size is not None:
        width, height = size
    else:
        width, height = (600, 600)
    fig = go.Figure(data=bar(x, y), layout=layout(title, width, height))
    iplot(fig)
    if file is not None:
        fig.write_image(f'reports/figures/{file}.png')


def plot_scatter(x, ys, labels, title, xaxis='', yaxis='', file=None):
    fig = go.Figure()
    _ = fig.update_layout(title=dict(text=title,
                                     font=dict(size=25)),
                          legend=dict(y=-0.3,
                                      yanchor='bottom',
                                      orientation='h',
                                      font=dict(size=18)
                                      ),
                          xaxis=dict(title=dict(
                              text=xaxis,
                              font=dict(size=20)
                          )),
                          yaxis=dict(title=dict(
                              text=yaxis,
                              font=dict(size=20)
                          ))
                          )
    for i in range(len(ys)):
        _ = fig.add_trace(go.Scatter(x=x, y=ys[i],
                                     mode='lines+markers',
                                     name=labels[i]))
    fig.show()
    if file is not None:
        fig.write_image(f'reports/figures/{file}.png')
