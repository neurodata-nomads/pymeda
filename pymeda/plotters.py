import os

from plotly.offline import iplot, plot
import plotly.graph_objs as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import colorlover as cl


def get_spaced_colors(n):
    max_value = 255
    interval = int(max_value / n)
    hues = range(0, max_value, interval)
    return cl.to_rgb(["hsl(%d,100%%,40%%)" % i for i in hues])


def get_heat_colors(n):
    max_value = 255
    interval = int(max_value / n)
    hues = range(0, max_value, interval)
    return cl.to_rgb(["hsl(%d,100%%,40%%)" % i for i in hues])


def get_plt_cmap(cmap, n):
    """
    Helper function that converts matplotlib cmap to 
    integers in R, G, B space.

    Parameters
    ----------
    cmap : str
        Colormap from matplotlib 
    n : int
        Number of colors to output
        
    Returns
    -------
    out : list
        List of RGB values in format that can be used in Plotly
    """
    ranges = np.linspace(0, 1, num=n)
    arr = plt.cm.get_cmap(cmap)(ranges)
    arr = arr[:, :3] * 255

    out = []
    for r, g, b in arr.astype(np.int):
        out.append('rgb({},{},{})'.format(r, g, b))

    return out


class MatrixPlotter:
    def __init__(self, DS, mode="notebook", base_path=None):
        self.DS = DS
        self.plot_mode = mode
        self.base_path = base_path

        Reds = cl.scales['8']['seq']['Reds']
        self.Reds = list(zip(np.linspace(0, 1, len(Reds)), Reds))

        BuRd = cl.scales['11']['div']['RdBu'][::-1]
        self.BuRd = list(zip(np.linspace(0, 1, len(BuRd)), BuRd))

    def makeplot(self, fig, local_path=None):
        """Make the plotly figure visable to the user in the way they want.

        Parameters
        ----------
        gid : :obj:`figure`
            An plotly figure.

        """
        if self.plot_mode == "notebook":
            iplot(fig)
        if self.plot_mode == "savediv":
            fig["layout"]["autosize"] = True
            div = plot(fig, output_type='div', include_plotlyjs=False)
            path = os.path.join(self.base_path, local_path + ".html")
            os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
            with open(path, "w") as f:
                f.write(div)
                f.close()

        if self.plot_mode == "div":
            fig["layout"]["autosize"] = True
            return plot(fig, output_type='div', include_plotlyjs=False)

    def _get_layout(self, title, xaxis, yaxis):
        if self.plot_mode == "div":
            return dict(xaxis=xaxis, yaxis=yaxis)
        else:
            return dict(title=title, xaxis=xaxis, yaxis=yaxis)


class Heatmap(MatrixPlotter):
    titlestring = "%s Heatmap"
    shortname = "heatmap"

    def plot(self, showticklabels=False):
        title = self.titlestring % (self.DS.name)
        xaxis = go.XAxis(
            title="Observations",
            ticktext=self.DS.D.index,
            ticks="",
            showticklabels=False,
            tickvals=[i for i in range(len(self.DS.D.index))])
        yaxis = go.YAxis(
            title="Dimensions",
            ticktext=self.DS.D.columns,
            ticks="",
            showticklabels=showticklabels,
            tickvals=[i for i in range(len(self.DS.D.columns))])
        layout = self._get_layout(title, xaxis, yaxis)

        maximum = self.DS.D.max().max()
        trace = go.Heatmap(
            z=self.DS.D.as_matrix().T,
            zmin=-maximum,
            zmax=maximum,
            colorscale=self.BuRd)
        data = [trace]
        fig = dict(data=data, layout=layout)
        return self.makeplot(fig, "agg/" + self.shortname)


class LocationHeatmap(MatrixPlotter):
    titlestring = "%s Location Heatmap"
    shortname = "locationheat"

    def plot(self, showticklabels=False):
        title = self.titlestring % (self.DS.name)
        D = self.DS.D.as_matrix().T
        means = np.mean(D, axis=1)
        medians = np.median(D, axis=1)
        z = np.vstack([means, medians])
        yaxis = go.YAxis(
            ticktext=["mean", "median"], showticklabels=True, tickvals=[0, 1])
        xaxis = go.XAxis(title="dimensions", showticklabels=showticklabels)
        layout = self._get_layout(title, xaxis, yaxis)

        trace = go.Heatmap(x=self.DS.D.columns, z=z, colorscale=self.Reds)
        data = [trace]
        fig = dict(data=data, layout=layout)
        return self.makeplot(fig, "agg/" + self.shortname)


class LocationLines(MatrixPlotter):
    titlestring = "%s Embedding Location Lines"
    shortname = "locationlines"

    def plot(self, showticklabels=False):
        title = self.titlestring % (self.DS.name)
        D = self.DS.D.as_matrix().T
        means = np.mean(D, axis=1)
        medians = np.median(D, axis=1)
        trace0 = go.Scatter(x=self.DS.D.columns, y=means, name="means")
        trace1 = go.Scatter(x=self.DS.D.columns, y=medians, name="medians")

        xaxis = dict(title="Dimensions", showticklabels=showticklabels)
        yaxis = dict(title="Mean or Median Value")
        layout = self._get_layout(title, xaxis, yaxis)

        data = [trace0, trace1]
        fig = dict(data=data, layout=layout)
        return self.makeplot(fig, "agg/" + self.shortname)


class HistogramHeatmap(MatrixPlotter):
    titlestring = "%s Histogram Heatmap"
    shortname = "histogramheat"

    def plot(self, showticklabels=False, scale=None):
        title = self.titlestring % (self.DS.name)
        D = self.DS.D.as_matrix().T
        d, n = D.shape
        D = (D - np.mean(D, axis=1).reshape(d, 1)) / np.std(
            D, axis=1).reshape(d, 1)
        D = np.nan_to_num(D)  # only nan if std all 0 -> all values 0

        num_bins = int(np.sqrt(2 * n))
        if num_bins > 20:
            num_bins = 20
        min_val = np.floor(np.min(D))
        if min_val < -5:
            min_val = -5
        max_val = np.ceil(np.max(D))
        if max_val > 5:
            max_val = 5
        bins = np.linspace(min_val, max_val,
                           (max_val - min_val) * num_bins + 1)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        H = []
        for i in range(D.shape[0]):
            hist = np.histogram(D[i, :], bins=bins)[0]
            H.append(hist)
        z = np.vstack(H).astype(np.float)

        if scale == 'log':
            z[z > 0] = np.log(z[z > 0], dtype=np.float)

        trace = go.Heatmap(
            y=self.DS.D.columns,
            z=z,
            x=bins,
            colorscale=self.Reds,
            colorbar=go.ColorBar(title='Counts'))
        data = [trace]
        xaxis = go.XAxis(
            title="Normalized Value",
            ticks="outside",
            showticklabels=True,
        )
        yaxis = go.YAxis(
            title="Dimensions",
            ticks="",
            showticklabels=showticklabels,
            mirror=True)
        layout = self._get_layout(title, xaxis, yaxis)
        fig = dict(data=data, layout=layout)
        return self.makeplot(fig, "agg/" + self.shortname)


class RidgeLine(MatrixPlotter):
    titlestring = "%s Ridgeline Plot"
    shortname = "ridgeline"

    def plot(self):
        title = self.titlestring % (self.DS.name)
        D = self.DS.D.as_matrix().T
        columns = self.DS.D.columns[::-1]
        d, n = D.shape
        # Standardize each feature so that mean=0, std=1
        D = (D - np.mean(D, axis=1).reshape(d, 1)) / np.std(
            D, axis=1).reshape(d, 1)
        D = np.nan_to_num(D)  # only nan if std all 0 -> all values 0

        # Get colors
        colors = get_plt_cmap('rainbow', d)

        # Clip the min and max values at -5 and 5 respectively
        min_val = np.floor(np.min(D))
        if min_val < -5:
            min_val = -5
        max_val = np.ceil(np.max(D))
        if max_val > 5:
            max_val = 5

        x_range = np.linspace(min_val, max_val, 100)

        # calculate guassian KDEs
        kdes = []
        for row in D:
            kde = stats.kde.gaussian_kde(row)
            kdes.append(kde(x_range))

        # Spacing between each ridgeline
        spacing = 0.5

        # Plot each ridgelines
        data = []
        for idx, y in enumerate(kdes[::-1]):
            y += idx * spacing  # Amount to separate each ridgeline
            trace = go.Scatter(
                x=x_range,
                y=y,
                name=columns[idx],
                mode='lines',
                line=dict(color='rgb(0,0,0)', width=1.5),
                fill='toself',
                fillcolor=colors[idx],
                opacity=.6)
            data.append(trace)

        # Controls placement of y-axis tick labels
        tickvals = np.arange(len(data)) * spacing
        yaxis = dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=True,
            tickmode='array',
            ticktext=columns,
            tickvals=tickvals,
            rangemode='nonnegative')

        xaxis = dict(
            showline=False,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            autotick=False,
            ticks='outside',
            tickcolor='rgb(204, 204, 204)')

        if self.plot_mode == "div":
            layout = go.Layout(
                showlegend=False,
                height=max(42 * len(data), 600),
                xaxis=xaxis,
                yaxis=yaxis)
        else:
            layout = go.Layout(
                showlegend=False,
                height=max(42 * len(data), 600),
                xaxis=xaxis,
                yaxis=yaxis,
                title=title)

        # Reverse order since lastest plot is on the front
        fig = go.Figure(data=data[::-1], layout=layout)
        return self.makeplot(fig, "agg/" + self.shortname)


class CorrelationMatrix(MatrixPlotter):
    titlestring = "%s Correlation Matrix"
    shortname = "correlation"

    def plot(self, showticklabels=False):
        title = self.titlestring % (self.DS.name)
        D = self.DS.D.as_matrix().T
        xaxis = dict(
            title="Dimensions",
            ticks="",
            showgrid=False,
            zeroline=False,
            showticklabels=showticklabels,
        )
        yaxis = dict(
            scaleanchor="x",
            title="Dimensions",
            ticks="",
            showgrid=False,
            zeroline=False,
            showticklabels=showticklabels,
        )
        layout = dict(title=title, xaxis=xaxis, yaxis=yaxis)
        with np.errstate(divide='ignore', invalid='ignore'):
            C = np.nan_to_num(np.corrcoef(D))

        layout = self._get_layout(title, xaxis, yaxis)
        trace = go.Heatmap(
            x=self.DS.D.columns,
            y=self.DS.D.columns,
            z=C,
            zmin=-1,
            zmax=1,
            colorscale=self.BuRd)
        fig = dict(data=[trace], layout=layout)
        return self.makeplot(fig, "agg/" + self.shortname)


class ScreePlotter(MatrixPlotter):
    titlestring = "%s Scree Plot"
    shortname = "scree"

    def plot(self):
        title = self.titlestring % (self.DS.name)
        D = self.DS.D.as_matrix().T
        _, S, _ = np.linalg.svd(D, full_matrices=False)
        y = S
        x = np.arange(1, len(S) + 1)
        sy = np.sum(y)
        cy = np.cumsum(y)
        xaxis = dict(title='Factors')
        yaxis = dict(title='Proportion of Total Variance')
        var = go.Scatter(mode='lines+markers', x=x, y=y / sy, name="Variance")
        cumvar = go.Scatter(
            mode='lines+markers', x=x, y=cy / sy, name="Cumulative Variance")
        data = [var, cumvar]
        layout = self._get_layout(title, xaxis, yaxis)
        fig = dict(data=data, layout=layout)
        return self.makeplot(fig, "agg/" + self.shortname)


class HierarchicalClusterMeansDendrogram(MatrixPlotter):
    titlestring = "%s %s Cluster Means Dendrogram, Level %d"
    shortname = "cmd"

    def plot(self):
        title = self.titlestring % (self.DS.name, self.DS.clustname,
                                    self.DS.levels)
        self.shortname = self.DS.shortclustname + self.shortname
        means = []
        for c in self.DS.clusters[self.DS.levels]:
            means.append(np.average(c, axis=0))
        X = np.column_stack(means).T
        try:
            fig = ff.create_dendrogram(X)
        except:
            return '''
                <div class="row" style="margin-top:20%">
                    <div class="col-md-4 offset-md-4 text-center">
                        <h1><b>Only one cluster found.</b></h1>
                        <h3>Perhaps try another algorithm?</h2>
                </div>
                '''
        if self.plot_mode != "div":
            fig["layout"]["title"] = title

        fig["layout"]["xaxis"]["title"] = "Cluster Labels"
        fig["layout"]["yaxis"]["title"] = "Cluster Mean Distances"
        del fig.layout["width"]
        del fig.layout["height"]
        return self.makeplot(fig, "agg/" + self.shortname)


class HierarchicalStackedClusterMeansHeatmap(MatrixPlotter):
    titlestring = "%s %s Stacked Cluster Means, Level %d"
    shortname = "scmh"

    def plot(self, showticklabels=False):
        title = self.titlestring % (self.DS.name, self.DS.clustname,
                                    self.DS.levels)
        self.shortname = self.DS.shortclustname + self.shortname

        Xs = []
        for l in self.DS.clusters[1:self.DS.levels + 1]:
            #When number of samples is too high, need to downsample
            freq = [c.shape[0] for c in l]
            if sum(freq) > 500:
                freq = [round((x / sum(freq)) * 500) for x in freq]
                if sum(freq) != 500:  #Rounding can give numbers not exactly 500
                    freq[freq.index(max(freq))] += (500 - sum(freq))

            means = []
            for i, c in enumerate(l):
                means += [np.average(c, axis=0)] * freq[i]

            X = np.column_stack(means)
            Xs.append(X)
        X = np.vstack(Xs)[::-1, :]
        y_labels = np.tile(self.DS.columns,
                           X.shape[0] // len(self.DS.columns))[::-1]
        trace = go.Heatmap(
            z=X, zmin=-np.max(X), zmax=np.max(X), colorscale=self.BuRd)
        data = [trace]
        xaxis = go.XAxis(
            title="Clusters",
            showticklabels=False,
            ticks="",
            mirror=True,
            tickvals=[i for i in range(X.shape[1])])
        yaxis = go.YAxis(
            title="Dimensions",
            showticklabels=showticklabels,
            ticks="",
            ticktext=y_labels,
            tickvals=[i for i in range(len(y_labels))],
            mirror=True)
        emb_size = len(np.average(self.DS.clusters[0][0], axis=0))
        bar_locations = np.arange(0, X.shape[0] + emb_size - 1, emb_size) - 0.5
        shapes = [
            dict(type="line", x0=-0.5, x1=X.shape[1] - 0.5, y0=b, y1=b)
            for b in bar_locations
        ]
        if self.plot_mode == "div":
            layout = dict(xaxis=xaxis, yaxis=yaxis, shapes=shapes)
        else:
            layout = dict(title=title, xaxis=xaxis, yaxis=yaxis, shapes=shapes)
        fig = dict(data=data, layout=layout)
        return self.makeplot(fig, "agg/" + self.shortname)


class ClusterMeansLevelHeatmap(MatrixPlotter):
    titlestring = "%s %s Cluster Means, Level %d"
    shortname = "cmlh"

    def plot(self, showticklabels=False):
        title = self.titlestring % (self.DS.name, self.DS.clustname,
                                    self.DS.levels)
        self.shortname = self.DS.shortclustname + self.shortname

        #When number of samples is too high, need to downsample
        freq = [c.shape[0] for c in self.DS.clusters[self.DS.levels]]
        if sum(freq) > 500:
            freq = [round((x / sum(freq)) * 500) for x in freq]

        means = []
        for i, c in enumerate(self.DS.clusters[self.DS.levels]):
            means += [np.average(c, axis=0)] * freq[i]
        X = np.column_stack(means)
        trace = go.Heatmap(
            y=self.DS.columns[::-1],
            z=np.flipud(X),
            zmin=-np.max(X),
            zmax=np.max(X),
            colorscale=self.BuRd)
        data = [trace]
        xaxis = go.XAxis(
            title="Clusters",
            showticklabels=False,
            ticks="",
            mirror=True,
            tickvals=[i for i in range(X.shape[1])])
        yaxis = go.YAxis(
            title="Dimensions",
            showticklabels=showticklabels,
            ticks="",
            mirror=True)
        layout = self._get_layout(title=title, xaxis=xaxis, yaxis=yaxis)
        fig = dict(data=data, layout=layout)
        return self.makeplot(fig, "agg/" + self.shortname)


class ClusterMeansLevelLines(MatrixPlotter):
    titlestring = "%s %s Cluster Means, Level %d"
    shortname = "cmll"

    def plot(self, showticklabels=False):
        title = self.titlestring % (self.DS.name, self.DS.clustname,
                                    self.DS.levels)
        self.shortname = self.DS.shortclustname + self.shortname
        data = []
        colors = get_spaced_colors(len(self.DS.clusters[self.DS.levels]))

        #When number of samples is too high, need to downsample
        freq = [c.shape[0] for c in self.DS.clusters[self.DS.levels]]
        if sum(freq) > 300:
            freq = [round((x / sum(freq)) * 300) for x in freq]

        for i, c in enumerate(self.DS.clusters[self.DS.levels]):
            data.append(
                go.Scatter(
                    x=np.average(c, axis=0),
                    y=self.DS.columns,
                    mode="lines",
                    line=dict(width=np.sqrt(freq[i]), color=colors[i]),
                    name="cluster " + str(i)))
        xaxis = go.XAxis(
            title="Mean Values", showticklabels=False, mirror=True)
        yaxis = go.YAxis(
            title="Dimensions", showticklabels=showticklabels, mirror=True)
        layout = self._get_layout(title=title, xaxis=xaxis, yaxis=yaxis)
        fig = dict(data=data, layout=layout)
        return self.makeplot(fig, "agg/" + self.shortname)


class ClusterPairsPlot(MatrixPlotter):
    titlestring = "%s %s Classification Pairs Plot, Level %d"
    shortname = "cpp"

    def plot(self):
        title = self.titlestring % (self.DS.name, self.DS.clustname,
                                    self.DS.levels)
        self.shortname = self.DS.shortclustname + self.shortname
        data = []
        colors = get_spaced_colors(len(self.DS.clusters[self.DS.levels]))
        samples = []
        labels = []
        for i, c in enumerate(self.DS.clusters[self.DS.levels]):
            samples.append(c.T)
            labels.append(c.shape[0] * [i])
        samples = np.hstack(samples)[:3, :]
        labels = np.hstack(labels)
        df = pd.DataFrame(
            samples.T, columns=["Dim %d" % i for i in range(samples.shape[0])])
        df["label"] = ["Cluster %d" % i for i in labels]
        fig = ff.create_scatterplotmatrix(
            df, diag='box', index="label", colormap=colors)
        if self.plot_mode != "div":
            fig["layout"]["title"] = title
        else:
            fig["layout"]["title"] = None
        del fig.layout["width"]
        del fig.layout["height"]
        return self.makeplot(fig, "agg/" + self.shortname)
