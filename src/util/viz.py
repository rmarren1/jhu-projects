from plotly.offline import iplot
import numpy as np
import pandas as pd
import cufflinks as cl

def heatmap_channels(data, num, title = '', xtitle='', ytitle='', ztitle='', T = True):
    full_ind = range(len(data))
    ind = np.random.choice(full_ind, num, replace=False)
    d = None
    for i in ind:
        if T:
            d = data[i].T
        else:
            d = data[i]
        df = pd.DataFrame(data= data[i])
        fig = df.iplot(asFigure=True,
                    kind='heatmap',
                    xTitle=xtitle,
                    yTitle=ytitle,
                    title=title)
        fig.data.update(dict(colorbar=dict(title=ztitle, titleside="right")))
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
