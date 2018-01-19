from itertools import chain, repeat
from functools import partial
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import pandas as pd
from intern.resource.boss.resource import ChannelResource
from lemur.plotters import HGMMPlotter

from neurodataresource import NeuroDataResource
from util import block_compute, read_csv
from wrappers import cached_property


class Meda(NeuroDataResource):
    """
    TODO: Write docs. Also open for suggestions for class name. 
    """

    def __init__(self,
                 host,
                 token,
                 collection,
                 experiment,
                 annotation_name,
                 threads=10):
        """
        Constructor

        Parameters
        ----------
        host : str
        token : str
            Boss API key
        collection : str
            Name of collection in Boss
        experiment : str
            Name of experiment within collection
        annotation_name : str
            Name of an annotation volume. Volume must be a uint64 image.
        """
        super().__init__(host, token, collection, experiment)

        #TODO: This is jank. Fix pls.
        self.annotation_resource = self._bossRemote.get_project(
            ChannelResource(annotation_name, collection, experiment))
        self.threads = threads

    @cached_property
    def labels(self):
        """
        Obtains all unique ids in an annotation volume.

        Returns
        -------
        ids : 1d-array like
            Array of integers
        """
        print('Fetching all unique annotation labels.')

        z_max, y_max, x_max = self.max_dimensions
        blocks = block_compute(
            0, x_max, 0, y_max, 0, z_max, block_size=(1024, 1024, 16))

        with ThreadPool(processes=self.threads) as tp:
            z_slices = [block[2] for block in blocks]
            y_slices = [block[1] for block in blocks]
            x_slices = [block[0] for block in blocks]

            args = zip(
                repeat(self.annotation_resource), repeat(0), x_slices,
                y_slices, z_slices)
            results = tp.starmap(self._bossRemote.get_ids_in_region, args)

        ids = np.unique(list(chain.from_iterable(results)))

        return ids

    @labels.setter
    def labels(self, csv_file):
        """
        Setter for unique labels
        """
        index_col = 'labels'

        return read_csv(csv_file, cols=index_col)

    @cached_property
    def bounds(self):
        """
        Obtains tight bounding boxes for each unique ids.

        Returns
        -------
        bounds : 2d-array like
            In format z_min, z_max, y_min, y_max, x_min, x_max order.
        """
        print('Fetching bounding box for each annotation label.')

        with ThreadPool(processes=self.threads) as tp:
            args = zip(
                repeat(self.annotation_resource), repeat(0), self.labels,
                repeat('tight'))
            results = tp.starmap(self._bossRemote.get_bounding_box, args)

        bounds = np.empty((len(results), 6), dtype=np.int)

        #TODO: Vectorize this shit

        for idx, result in enumerate(results):
            z_min, z_max = result['z_range']
            y_min, y_max = result['y_range']
            x_min, x_max = result['x_range']
            bounds[idx, :] = z_min, z_max, y_min, y_max, x_min, x_max

        return bounds

    @cached_property
    def centroids(self):
        """
        Calculates centroids. Downloads cutout of annotation channel for each bounding 
        box then finds the volumetric centroid of the cutout.

        Returns
        -------
        centroids : 2d-array
            Array of each centroids in z, y, x format
        """
        print('Calculating centroids.')
        z_ranges = self.bounds[:, :2]
        y_ranges = self.bounds[:, 2:4]
        x_ranges = self.bounds[:, 4:]

        with ThreadPool(processes=self.threads) as tp:
            args = zip(
                repeat(self.annotation_resource), repeat(0), x_ranges,
                y_ranges, z_ranges)
            cutouts = tp.starmap(self._bossRemote.get_cutout, args)

        centroids = np.empty((len(self.bounds), 3))
        for idx, cutout in enumerate(cutouts):
            z, y, x = np.nonzero(cutout == self.labels[idx])
            centroids[idx, :] = np.mean(z) + z_ranges[idx][0], np.mean(
                y) + y_ranges[idx][0], np.mean(x) + x_ranges[idx][0]

        return centroids.astype(np.int)

    @centroids.setter
    def centroids(self, csv_file):
        """
        Setter for centroids based on csv file. CSV file must have z, y, x
        columns.

        Parameters
        ----------
        csv_file : str
            Path to the csv file. Must have z, y, x columns. 
        """
        cols = ['z', 'y', 'x']

        return read_csv(csv_file, cols=cols)

    def _initialize_properties(self):
        """
        Helper function to initialize labels, bounds, and centroids 
        properties, which are lazily evaluated.
        """
        self.labels
        self.bounds
        self.centroids

    def calculate_dimensions(self, size):
        """
        Calculates the coordinates of a box surrounding
        the centroid with size (z, y, x).

        Parameters
        ----------
        centroids : 2d-array like
            Input point with order (z, y, x)
        size : 1d-array like
            Shape of box surrounding centroid with order (z, y, x)
        """
        z_dim, y_dim, x_dim = [(i - 1) // 2 for i in size]
        z_max, y_max, x_max = self.max_dimensions

        grid = np.array([-z_dim, z_dim, -y_dim, y_dim, -x_dim, x_dim])

        dimensions = np.repeat(self.centroids, 2, axis=1) + grid

        np.clip(
            dimensions[:, 0:2], a_min=0, a_max=z_max, out=dimensions[:, 0:2])
        np.clip(
            dimensions[:, 2:4], a_min=0, a_max=y_max, out=dimensions[:, 2:4])
        np.clip(
            dimensions[:, 4:6], a_min=0, a_max=x_max, out=dimensions[:, 4:6])

        return dimensions

    def calculate_stats(self, channels, size=None, pyfunc=np.mean, mask=None):
        """
        Calculates aggregate based on pyfunc specified by the user. Defaults 
        to average intensity. 

        Parameters
        ----------
        channels : list of str
            List of channels to calculate F0. Specify order.
        size : 1d-array like, optional
            If None, use tight bounds. Otherwise, size of boxes to build around 
            centroid in (z, y, x) format.
        pyfunc : callable
            A python function or method (e.g. np.mean).
        mask : optional

        TODO: Gaussian masking
        """
        #Get labels, bounds, and centroids
        self._initialize_properties()

        if size:
            dimensions = self.calculate_dimensions(size)
        else:
            dimensions = self.bounds

        data = np.empty((len(dimensions), len(channels)))

        for idx, channel in enumerate(channels):
            print('Calculating features on {}'.format(channel))
            with ThreadPool(processes=self.threads) as tp:
                args = zip(
                    repeat(channel), dimensions[:, 0:2], dimensions[:, 2:4],
                    dimensions[:, 4:6])
                cutouts = tp.starmap(self.get_cutout, args)
                temp = list(map(pyfunc, cutouts))
                data[:, idx] = temp
                """
                if size:
                    data[:, idx] = np.array(list(map(pyfunc, results)))
                else:
                    data[:, idx] = np.divide(
                        list(map(np.sum, np.multiply(results, masks))),
                        list(map(np.sum, masks)))"""
                #data[:, i] = np.multiply(np.array(list(map(np.sum, results))), masks)

        df = pd.DataFrame(data, index=self.labels, columns=channels)
        return df

    def gaussian_mask():
        #TODO: write guassian masking using FWHM
        pass

    def upload_to_boss(self, volume, host, token, channel_name, collection,
                       experiment):
        NeuroDataResource.ingest_volume(host, token, channel_name, collection,
                                        experiment, volume)

        print('You can view the upload here: ')

        url = 'https://ndwebtools.neurodata.io/ndviz_url/{}/{}/{}'.format(
            collection, experiment, channel_name)
        print(url)

    def create_cluster_vol(self, ds, levels, seed):
        out = np.empty(self.max_dimensions, dtype=np.uint64)
        hgmm = HGMMPlotter(ds, levels=levels, random_state=seed)

        l = []
        for cluster in hgmm.levels[levels]:
            temp = []
            for row in cluster[0]:
                temp.append(ds.D[ds.D.isin(row)].dropna().index[0])
            l.append(temp)

        for idx, cluster in enumerate(l):
            for label in cluster:
                index = np.where(self.labels == label)[0][0]
                z, y, x = self.dimensions[index]
                out[z[0]:z[1], y[0]:y[1], x[0]:x[1]] = levels * 10 + idx

        return out

    def export_data(self):
        return pd.concat([self.df, self.centroids], axis=1)