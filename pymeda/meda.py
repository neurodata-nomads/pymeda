import numpy as np
import pandas as pd
import lemur.datasets as lds
import lemur.plotters as lpl
import lemur.clustering as lcl
import knor

from scipy.stats import zscore
from wrappers import cached_property


class Meda:
    """
    TODO: Write docs. 
    """

    def __init__(self,
                 csv_file,
                 index_col=None,
                 title=None,
                 cluster_levels=1,
                 showticklabels=True,
                 mode='notebook'):
        """
        Parameters
        ----------
        csv_file : str
            Path to the csv_file to plot
        index_col : str, optional
            Name of the index column in csv_file
        title : str, optional
            Title of the dataset.
        cluster_levels : int (default=1)
            Number of levels for hierarchial clustering
        showticklabels : bool
            Default True
        mode : str
            notebook or div
        """
        self._ds = lds.CSVDataSet(csv_file, name=title, index_column=index_col)
        self._ds_normed = lds.DataSet(self._ds.D.apply(zscore), name=title)
        self.showticklabels = showticklabels
        self.cluster_levels = cluster_levels
        self.mode = mode
        self._cluster_ds = None

    def scree_plot(self):
        """
        Genereate scree plot of PCA results
        """
        return lpl.ScreePlotter(self._ds_normed, mode=self.mode).plot()

    def correlation_matrix(self):
        """
        Correlation matrix 
        """
        return lpl.CorrelationMatrix(
            self._ds_normed,
            mode=self.mode).plot(showticklabels=self.showticklabels)

    def _compute_clusters(self, cluster_mode=None):
        """
        Run hierarchial GMM clustering

        Parameters
        ----------
        cluster_mode : str 
            Not implemented yet
        """
        cluster_ds = lcl.HGMMClustering(
            self._ds_normed, levels=self.cluster_levels)
        self._cluster_ds = cluster_ds

    def cluster_dendrogram(self):
        if not self._cluster_ds:
            self._compute_clusters()

        return lpl.HGMMClusterMeansDendrogram(
            self._cluster_ds, mode=self.mode).plot()

    def cluster_means_stacked(self):
        """
        Generate hierarchial clustering stacked means plot
        """
        if not self._cluster_ds:
            self._compute_clusters()

        return lpl.HGMMStackedClusterMeansHeatmap(
            self._cluster_ds,
            mode=self.mode).plot(showticklabels=self.showticklabels)

    def cluster_means_heatmap(self):
        """
        Generate hierarchial cluster means plot of the lowest level
        """
        if not self._cluster_ds:
            self._compute_clusters()

        return lpl.HGMMClusterMeansLevelHeatmap(
            self._cluster_ds,
            mode=self.mode).plot(showticklabels=self.showticklabels)

    def cluster_means_lines(self):
        """
        Generate hierarchial cluster mean lines plot of the lowest level
        """
        if not self._cluster_ds:
            self._compute_clusters()

        return lpl.HGMMClusterMeansLevelLines(
            self._cluster_ds,
            mode=self.mode).plot(showticklabels=self.showticklabels)

    def cluster_pair_plot(self):
        """
        Generate pair plot for clusters
        """
        if not self._cluster_ds:
            self._compute_clusters()

        return lpl.HGMMPairsPlot(self._cluster_ds, mode=self.mode).plot()

    def location_lines(self):
        """
        Generate location estimates using line graph
        """
        return lpl.LocationLines(
            self._ds, mode=self.mode).plot(showticklabels=self.showticklabels)

    def location_heatmap(self):
        """
        Generate location estimates using heatmap
        """
        return lpl.LocationHeatmap(
            self._ds, mode=self.mode).plot(showticklabels=self.showticklabels)

    def d1_heatmap(self):
        """
        Generate 1d heatmaps
        """
        if self._ds.n > 1000:  #if sample size is > 1000, run kmeans++ initialization
            ret = knor.Kmeans(
                self._ds_normed.D.values, 1000, max_iters=0, init='kmeanspp')
            centroids_df = pd.DataFrame(
                ret.get_centroids(), columns=self._ds.D.columns)
            centroids_ds = lds.DataSet(centroids_df, name=self._ds.name)

            return lpl.HistogramHeatmap(
                centroids_ds,
                mode=self.mode).plot(showticklabels=self.showticklabels)
        else:
            return lpl.HistogramHeatmap(
                self._ds_normed,
                mode=self.mode).plot(showticklabels=self.showticklabels)

    def run_all(self):
        pass