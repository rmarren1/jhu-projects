from plotly.offline import iplot
import numpy as np
import pandas as pd
import cufflinks as cl
#cl.go_offline()

def heatmap_channels(data, title = '', xtitle='', ytitle='', ztitle='', T = True):
    if T:
        data = data.T
    df = pd.DataFrame(data= data)
    fig = df.iplot(asFigure=True,
                kind='heatmap',
                xTitle=xtitle,
                colorscale='RdBu')
    fig.data.update(dict(transpose=True, 
        colorbar=dict(title=ztitle, titleside="right")))
    fig.layout.update(dict(title=title,
                           height=600,
                           width=600,
                           yaxis=dict(title=ytitle,
                                      autorange='reversed')))
    iplot(fig)


def line(data, num, title = '', xtitle='', ytitle='', ztitle=''):
    full_ind = range(len(data))
    ind = np.random.choice(full_ind, num, replace=False)
    for i in ind:
        df = pd.DataFrame(data = data[i])
        fig = df.iplot(asFigure=True,
                        kind='line',
                        xTitle=xtitle,
                        yTitle=ytitle,
                        title=title)
        iplot(fig)
