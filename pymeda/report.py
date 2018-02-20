import argparse
import time

import matplotlib as mpl
mpl.use('TkAgg')
import lemur.datasets as lds
import lemur.plotters as lpl
import lemur.clustering as lcl
import pandas as pd

from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from scipy.stats import zscore


def make_plots(csv_file, title=None, index_col=None):
    ds = lds.CSVDataSet(csv_file, name=title, index_column=index_col)

    #Make plots and save as div
    location_heatmap = lpl.LocationHeatmap(
        ds, mode='div').plot(showticklabels=True)
    location_lines = lpl.LocationLines(
        ds, mode='div').plot(showticklabels=True)

    ds.D = ds.D.apply(zscore)

    histogram_heatmap = lpl.HistogramHeatmap(
        ds, mode='div').plot(showticklabels=True)
    scree_plot = lpl.ScreePlotter(ds, mode='div').plot()
    corr_matrix = lpl.CorrelationMatrix(
        ds, mode='div').plot(showticklabels=True)

    #HGMM plots
    hgmm_ds = lcl.HGMMClustering(ds, levels=1)
    hgmm_dendogram = lpl.HGMMClusterMeansDendrogram(hgmm_ds, mode='div').plot()
    hgmm_pair_plot = lpl.HGMMPairsPlot(hgmm_ds, mode='div').plot()
    hgmm_stacked_mean = lpl.HGMMStackedClusterMeansHeatmap(
        hgmm_ds, mode='div').plot(showticklabels=True)
    hgmm_cluster_means = lpl.HGMMClusterMeansLevelLines(
        hgmm_ds, mode='div').plot(showticklabels=True)

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


def parse_cmd_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--file_name',
        type=str,
        help='Input file in csv or comma delimited format',
        required=True)
    parser.add_argument(
        '-i', '--index_col', type=str, help='Index column in input file')
    parser.add_argument(
        '-t', '--title', type=str, help='Title of the datatset')
    parser.add_argument(
        '-o',
        '--outdir',
        type=str,
        help='Path to output directory',
        required=True)

    result = parser.parse_args()
    return result


if __name__ == '__main__':
    date = time.strftime("%Y-%m-%d")

    result = parse_cmd_line_args()
    plots = make_plots(
        result.file_name, title=result.title, index_col=result.index_col)
    report = generate_report(plots, title=result.title, date=date)

    if result.outdir.endswith('/'):
        outdir = result.outdir + date + '_' + result.title + '.html'
    else:
        outdir = result.outdir + '/' + date + '_' + result.title + '.html'

    with open(outdir, 'w') as f:
        f.write(report)

    print('Report saved at {}'.format(outdir))