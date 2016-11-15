from plotly.offline import iplot
import numpy as np
import pandas as pd
import cufflinks as cl

def heatmap_channels(data, num, title = '', xtitle='', ytitle='', ztitle=''):
    full_ind = range(len(data))
    ind = np.random.choice(full_ind, num, replace=False)
    for i in ind:
        df = pd.DataFrame(data= data[i].T)
        fig = df.iplot(asFigure=True,
                    kind='heatmap',
                    xTitle=xtitle,
                    yTitle=ytitle,
                    title=title,
                    colorscale='blues')
        fig.data.update(dict(colorbar=dict(title=ztitle, titleside="right")))
        iplot(fig)
