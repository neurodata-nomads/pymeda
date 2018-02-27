import pkg_resources
import time

import numpy as np
import pandas as pd
import lemur.datasets as lds
import lemur.plotters as lpl
import lemur.clustering as lcl
import knor

from scipy.stats import zscore
from .wrappers import cached_property

from plotly.offline import (download_plotlyjs, init_notebook_mode)
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader


class Meda:
    """
    TODO: Write docs. 
    """

    def __init__(self,
                 csv_file,
                 title,
                 index_col=None,
                 cluster_levels=1,
                 showticklabels=True,
                 mode='notebook'):
        """
        Parameters
        ----------
        csv_file : str
            Path to the csv_file to plot
        title : str
            Title of the dataset.
        index_col : str, optional
            Name of the index column in csv_file
        cluster_levels : int (default=1)
            Number of levels for hierarchial clustering
        showticklabels : bool
            Default True
        mode : str
            notebook or div
        """
        self._ds = lds.CSVDataSet(csv_file, name=title, index_column=index_col)
        self._ds_normed = lds.DataSet(self._ds.D.apply(zscore), name=title)
        self._showticklabels = showticklabels
        self._cluster_levels = cluster_levels
        self._mode = mode
        if mode == 'notebook':
            init_notebook_mode()
        self._cluster_ds = None

    def d1_heatmap(self, mode=None):
        """
        Generate 1d heatmaps
        """
        if not mode:
            mode = self.mode

        if self._ds.n > 1000:  #if sample size is > 1000, run kmeans++ initialization
            ret = knor.Kmeans(
                self._ds_normed.D.values, 1000, max_iters=0, init='kmeanspp')
            centroids_df = pd.DataFrame(
                ret.get_centroids(), columns=self._ds.D.columns)
            centroids_ds = lds.DataSet(centroids_df, name=self._ds.name)

            return lpl.HistogramHeatmap(
                centroids_ds,
                mode=mode).plot(showticklabels=self._showticklabels)
        else:
            return lpl.HistogramHeatmap(
                self._ds_normed,
                mode=mode).plot(showticklabels=self._showticklabels)

    def scree_plot(self, mode=None):
        """
        Genereate scree plot of PCA results
        """
        if not mode:
            mode = self.mode

        return lpl.ScreePlotter(self._ds_normed, mode=mode).plot()

    def correlation_matrix(self, mode=None):
        """
        Correlation matrix 
        """
        if not mode:
            mode = self.mode

        return lpl.CorrelationMatrix(
            self._ds_normed,
            mode=mode).plot(showticklabels=self._showticklabels)

    def _compute_clusters(self, cluster_mode=None):
        """
        Run hierarchial GMM clustering

        Parameters
        ----------
        cluster_mode : str 
            Not implemented yet
        """
        cluster_ds = lcl.HGMMClustering(
            self._ds_normed, levels=self._cluster_levels)
        self._cluster_ds = cluster_ds

    def cluster_dendrogram(self, mode=None):
        if not mode:
            mode = self._mode

        if not self._cluster_ds:
            self._compute_clusters()

        return lpl.HGMMClusterMeansDendrogram(
            self._cluster_ds, mode=mode).plot()

    def cluster_means_stacked(self, mode=None):
        """
        Generate hierarchial clustering stacked means plot
        """
        if not mode:
            mode = self._mode

        if not self._cluster_ds:
            self._compute_clusters()

        return lpl.HGMMStackedClusterMeansHeatmap(
            self._cluster_ds,
            mode=mode).plot(showticklabels=self._showticklabels)

    def cluster_means_heatmap(self, mode=None):
        """
        Generate hierarchial cluster means plot of the lowest level
        """
        if not mode:
            mode = self._mode

        if not self._cluster_ds:
            self._compute_clusters()

        return lpl.HGMMClusterMeansLevelHeatmap(
            self._cluster_ds,
            mode=mode).plot(showticklabels=self._showticklabels)

    def cluster_means_lines(self, mode=None):
        """
        Generate hierarchial cluster mean lines plot of the lowest level
        """
        if not mode:
            mode = self._mode

        if not self._cluster_ds:
            self._compute_clusters()

        return lpl.HGMMClusterMeansLevelLines(
            self._cluster_ds,
            mode=mode).plot(showticklabels=self._showticklabels)

    def cluster_pair_plot(self, mode=None):
        """
        Generate pair plot for clusters
        """
        if not mode:
            mode = self._mode

        if not self._cluster_ds:
            self._compute_clusters()

        return lpl.HGMMPairsPlot(self._cluster_ds, mode=mode).plot()

    def location_lines(self, mode=None):
        """
        Generate location estimates using line graph
        """
        if not mode:
            mode = self._mode

        return lpl.LocationLines(
            self._ds, mode=mode).plot(showticklabels=self._showticklabels)

    def location_heatmap(self, mode=None):
        """
        Generate location estimates using heatmap
        """
        if not mode:
            mode = self._mode

        return lpl.LocationHeatmap(
            self._ds, mode=mode).plot(showticklabels=self._showticklabels)

    def run_all(self, mode=None):
        if not mode:
            mode = self._mode

        #Make plots and save as div
        histogram_heatmap = self.d1_heatmap(mode=mode)
        location_heatmap = self.location_heatmap(mode=mode)
        location_lines = self.location_lines(mode=mode)

        scree_plot = self.scree_plot(mode=mode)
        corr_matrix = self.correlation_matrix(mode=mode)

        #HGMM plots
        hgmm_dendogram = self.cluster_dendrogram(mode=mode)
        if not self._ds.n > 10000:
            hgmm_pair_plot = self.cluster_pair_plot(mode=mode)
        hgmm_stacked_mean = self.cluster_means_stacked(mode=mode)
        hgmm_cluster_mean = self.cluster_means_heatmap(mode=mode)
        hgmm_cluster_means = self.cluster_means_lines(mode=mode)

        if mode == 'div':
            out = {
                "Histogram Heatmap": histogram_heatmap,
                "Location Heatmap": location_heatmap,
                "Location Lines": location_lines,
                "Scree Plot": scree_plot,
                "Correlation Matrix": corr_matrix,
                "Hierarchical GMM Dendogram": hgmm_dendogram,
                "Pair Plot": hgmm_pair_plot,
                "Cluster Stacked Means": hgmm_stacked_mean,
                "Cluster Mean Heatmap": hgmm_cluster_mean,
                "Cluster Mean Lines": hgmm_cluster_means
            }
            return out

    def generate_report(self, out):
        """
        Creates a report

        Parameters
        ----------
        out : str
            Path to output the report
        """
        #Generate plots
        mode = 'div'
        plots = self.run_all(mode=mode)

        #Variables for output report name
        date = time.strftime("%Y-%m-%d")
        title = self._ds.name

        #Using package directory
        path = '/templates/'
        template_path = pkg_resources.resource_filename(__name__, path)
        print(template_path)

        env = Environment(loader=FileSystemLoader([template_path]))
        template = env.get_template('report.html')
        report = template.render(title=title, plots=plots, date=date)

        if out.endswith('/'):
            out = out + date + '_' + title + '.html'
        else:
            out = out + '/' + date + '_' + title + '.html'

        with open(out, 'w') as f:
            f.write(report)

        print('Report saved at {}'.format(out))