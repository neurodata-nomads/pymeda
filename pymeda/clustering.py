from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.mixture import GaussianMixture


class Clustering(metaclass=ABCMeta):
    def __init__(self, DS, levels=1, random_state=None):
        self.DS = DS
        self.name = DS.name
        self.columns = DS.D.columns
        self.X = self.DS.D.as_matrix()
        self.levels = levels
        self.random_state = random_state
        self.clusters = []

    @abstractmethod
    def cluster(self):
        pass


class HGMMClustering(Clustering):
    def __init__(self, DS, levels=1, random_state=None):
        """
        Parameters
        ----------
        DS :obj:`Dataset`
        level : int
            Number of levels to cluster
        random_state : int (optional)
            Initialize Gaussian Mixture Model with specified random state
        """
        Clustering.__init__(self, DS, levels, random_state)
        self.clustname = 'HGMM'
        self.shortclustname = 'hgmm'

    def cluster(self):
        clusters = []
        n = self.X.shape[0]
        l0 = self.hgmml0(self.X, self.random_state)
        clusters.append(l0)
        li = self.gmmBranch(l0[0], self.random_state)
        clusters.append(li)
        while (len(li) < n) and (len(clusters) - 1 < self.levels):
            lip = []
            for c in li:
                q = self.gmmBranch(c, self.random_state)
                if q is not None:
                    lip.extend(q)
            clusters.append(lip)
            li = lip

        self.clusters = [list(map(lambda x: x[0], c)) for c in clusters]
        self.hierarch = clusters

    def gmmBranch(self, level, random_state):
        X, p, mu = level

        #Check BIC to see to split node
        gmm_1 = GaussianMixture(n_components=1, random_state=random_state)
        gmm_1.fit(X)
        bic_1 = gmm_1.bic(X)

        if len(X) != 0:  #Does not run when input has one sample point
            gmm_2 = GaussianMixture(n_components=2, random_state=random_state)
            gmm_2.fit(X)
            bic_2 = gmm_2.bic(X)
        else:
            bic_2 = bic_1

        if bic_2 < bic_1:
            X0 = X[gmm_2.predict(X) == 0, :]
            X1 = X[gmm_2.predict(X) == 1, :]
            mypro = np.rint(gmm_2.weights_ * p)
            return [(
                X0,
                int(mypro[0]),
                gmm_2.means_[0, :],
            ), (
                X1,
                int(mypro[1]),
                gmm_2.means_[1, :],
            )]
        else:
            return [(
                X,
                int(np.rint(p * gmm_1.weights_[0])),
                gmm_1.means_[0, :],
            )]

    def hgmml0(self, X, random_state):
        gmm = GaussianMixture(n_components=1, random_state=random_state)
        gmm.fit(X)
        return [(
            X,
            int(np.rint(X.shape[0] * gmm.weights_[0])),
            gmm.means_[0, :],
        )]
