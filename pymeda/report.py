import time

import matplotlib as mpl
mpl.use('TkAgg')
import lemur.datasets as lds
import lemur.plotters as lpl
import pandas as pd

from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from scipy.stats import zscore


def make_plots(csv_file, title=None, index_col=None):
    df = pd.read_csv(csv_file, index_col=index_col)

    ds = lds.DataSet(df, name=title)
    ds_normed = lds.DataSet(df.apply(zscore), name=title)

    #Make plots and save as div
    location_heatmap = lpl.LocationHeatmap(
        ds, mode='div').plot(showticklabels=True)
    location_lines = lpl.LocationLines(
        ds, mode='div').plot(showticklabels=True)
    histogram_heatmap = lpl.HistogramHeatmap(
        ds_normed, mode='div').plot(showticklabels=True)
    scree_plot = lpl.ScreePlotter(ds_normed, mode='div').plot()
    corr_matrix = lpl.CorrelationMatrix(
        ds_normed, mode='div').plot(showticklabels=True)

    #HGMM plots
    seed = 2132
    hgmm_dendogram = lpl.HGMMClusterMeansDendrogram(
        ds_normed, mode='div').plot(level=1)
    hgmm_pair_plot = lpl.HGMMPairsPlot(ds_normed, mode='div').plot(level=1)
    hgmm_stacked_mean = lpl.HGMMStackedClusterMeansHeatmap(
        ds_normed, mode='div').plot(
            level=2, showticklabels=True)
    hgmm_cluster_means = lpl.HGMMClusterMeansLevelLines(
        ds_normed, mode='div').plot(
            level=1, showticklabels=True)

    out = {
        "Location Heatmap": location_heatmap,
        "Location Lines": location_lines,
        "Histogram Heatmap": histogram_heatmap,
        "Scree Plot": scree_plot,
        "Correlation Matrix": corr_matrix,
        "Hierarchical GMM Dendogram": hgmm_dendogram,
        "Pair Plot": hgmm_pair_plot,
        "Stacked Means": hgmm_stacked_mean,
        "Cluster Means": hgmm_cluster_means
    }

    return out


def generate_report(plots, title, date):
    """
    Generate an html report using csv_file.
    """
    env = Environment(loader=FileSystemLoader(["./templates"]))
    template = env.get_template('report.html')
    result = template.render(title=title, plots=plots, date=date)

    return result


if __name__ == '__main__':
    title = 'Collman15 Tight Annotations'
    date = time.strftime("%Y-%m-%d")
    plots = make_plots('../data/collman15v2_tight_mean.csv', title=title)
    result = generate_report(plots, title=title, date=date)
    with open('output.html', 'w') as f:
        f.write(result)