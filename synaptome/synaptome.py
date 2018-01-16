from itertools import repeat
from functools import partial
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import lemur.plotters as lpl
import numpy as np
import pandas as pd
from intern.resource.boss.resource import ChannelResource
from scipy.sparse import csr_matrix
from scipy.stats import zscore

from neurodataresource import NeuroDataResource


class Synaptome(NeuroDataResource):
    """
    TODO: Write docs. Also open for suggestions for class name. 
    """

    def __init__(self, host, token, collection, experiment, annotation_name, threads=10):
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

        self.annotation_resource = self._bossRemote.get_project(
            ChannelResource(annotation_name, collection, experiment))
        self.labels = self.get_labels()
        self.bounds = self.get_bounds(threads=threads)
        self.centroids = self.get_centroids(threads=threads)
        #self.sp_arr = None

    def get_labels(self):
        """
        Obtains all unique ids in an annotation volume.

        Returns
        -------
        ids : list
            List of integers
        """
        print('Fetching all unique annotation labels.')

        z_max, y_max, x_max = self.max_dimensions
        ids = self._bossRemote.get_ids_in_region(
            self.annotation_resource, 0, [0, x_max], [0, y_max], [0, z_max])

        return ids

    def get_bounds(self, threads=10):
        """
        Obtains loose bounding boxes for each unique ids.

        Returns
        -------
        out : pandas dataframe
        """
        print('Fetching bounding box for each annotation label.')

        with ThreadPool(processes=threads) as tp:
            args = zip(
                repeat(self.annotation_resource), repeat(0), self.labels, repeat('tight'))
            bounds = tp.starmap(self._bossRemote.get_bounding_box, args)

        return bounds

    def get_centroids(self, threads=10):
        """
        Calculates centroids. Downloads cutout of each bounding box then finds the 
        volumetric centroid of the cutout.
        """
        print('Calculating centroids.')
        x_ranges = [bound['x_range'] for bound in self.bounds]
        y_ranges = [bound['y_range'] for bound in self.bounds]
        z_ranges = [bound['z_range'] for bound in self.bounds]

        with ThreadPool(processes=threads) as tp:
            args = zip(repeat(self.annotation_resource), repeat(0), x_ranges, y_ranges, z_ranges)
            cutouts = tp.starmap(self._bossRemote.get_cutout, args)

        #TODO: calculate centroids for each cutout
        self.cutouts = cutouts

    def F0(self,
           annotation_channel,
           channel_list,
           method='tight',
           size=None,
           mask=None):
        """
        Calculates integrated sum for given box size built around each centroids

        Parameters
        ----------
        annotation_channel : str
            Name of an annotation channel
        channel_list : list of str
            List of channels to calculate F0. Specify order.
        method : str
            'tight' builds tight bounds on annotation or 'box' builds box of user
            specified size.
        size : 1d-array like, (optional if method is 'tight')
            Size of boxes to build around centroid in (z, y, x) format.
        mask : optional

        TODO: Gaussian masking
        """
        if self.sp_arr == None:
            self.sp_arr, self.labels = self._get_sparse_annotation(
                annotation_channel)

        if method == 'box':
            self.centroids = self._calculate_centroids()
            self.dimensions = self._calculate_dimensions(
                self.centroids, size, self.max_dimensions)
        elif method == 'tight':
            self.dimensions = self._calculate_tight_bounds()
            with ThreadPool(processes=8) as tp:
                func = partial(self._resource.get_cutout, annotation_channel)
                results = tp.starmap(func, self.dimensions)
                masks = []
                for i, label in enumerate(self.labels):
                    masks.append(results[i] == label)

        data = np.empty((len(self.dimensions), len(channel_list)))

        for i, channel in enumerate(channel_list):
            print('Calculating features on {}'.format(channel))
            with ThreadPool(
                    processes=8) as tp:  #optimum number of connections is 8
                func = partial(self._resource.get_cutout, channel)
                results = tp.starmap(func, self.dimensions)
                if method == 'box':
                    data[:, i] = np.array(list(map(np.sum, results)))
                elif method == 'tight':
                    data[:, i] = np.divide(
                        list(map(np.sum, np.multiply(results, masks))),
                        list(map(np.sum, masks)))
                    #data[:, i] = np.multiply(np.array(list(map(np.sum, results))), masks)

        df = pd.DataFrame(data, index=self.labels, columns=channel_list)
        self.df = df
        return df


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
        hgmm = lpl.HGMMPlotter(ds, levels=levels, random_state=seed)

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

    def _get_sparse_annotation(self, annotation_channel):
        """
        TODO: if annotation is image volume, run connected components
        """
        print('Downloading annotation channel')
        img = self._resource.get_cutout(
            annotation_channel, [0, self.max_dimensions[0]],
            [0, self.max_dimensions[1]], [0, self.max_dimensions[2]])

        sp_arr = csr_matrix(img.reshape(1, -1))
        labels = np.unique(sp_arr.data)

        return sp_arr, labels

    def _calculate_tight_bounds(self):
        """
        Calculates the tight bound box for each object in sparse matrix
        """

        out = np.empty((len(self.labels), 3, 2), dtype=np.int)

        for i, label in enumerate(self.labels):
            z, y, x = np.unravel_index(
                self.sp_arr.indices[self.sp_arr.data == label],
                self.max_dimensions)

            zmin, zmax = np.min(z), np.max(z)
            ymin, ymax = np.min(y), np.max(y)
            xmin, xmax = np.min(x), np.max(x)

            out[i, :, :] = np.asarray([[zmin, zmax + 1], [ymin, ymax + 1],
                                       [xmin, xmax + 1]])

        return out

    def _calculate_centroids(self):
        """
        Calculates the volumetric centroid of a sparsely labeled
        image volume.
        """
        centroids = np.empty((len(self.labels), 3))

        for i, label in enumerate(self.labels):
            z, y, x = np.unravel_index(
                self.sp_arr.indices[self.sp_arr.data == label],
                self.max_dimensions)

            centroids[i] = np.mean(z), np.mean(y), np.mean(x)

        return centroids.astype(np.int)

    def _calculate_dimensions(self, centroids, size, max_dimensions):
        """
        Calculates the coordinates of a box surrounding
        the centroid with size (z, y, x).

        Parameters
        ----------
        centroids : 2d-array like
            Input point with order (z, y, x)
        size : 1d-array like
            Shape of box surrounding centroid with order (z, y, x)
        max_dimensions : 1d-array like
            Maximum dimensions of the data with order (z, y, x)
        """
        z_dim, y_dim, x_dim = [(i - 1) // 2 for i in size]
        z_max, y_max, x_max = max_dimensions

        grid = np.array([
            [-z_dim, z_dim],  #use np.tile here with [-1, 1]
            [-y_dim, y_dim],
            [-x_dim, x_dim]
        ])

        out = np.empty((len(centroids), 3, 2), dtype=np.int)

        for i, centroid in enumerate(centroids.astype(np.int)):
            out[i, :, :] = grid + centroid.reshape((-1, 1))

        np.clip(out[:, 0, :], a_min=0, a_max=z_max, out=out[:, 0, :])
        np.clip(out[:, 1, :], a_min=0, a_max=y_max, out=out[:, 1, :])
        np.clip(out[:, 2, :], a_min=0, a_max=x_max, out=out[:, 2, :])

        return out
