import os
import json
import statistics

import pandas as pd
import numpy as np


class DataSet:
    def __init__(self, D, name="default"):
        self.D = D
        self.n, self.d = self.D.shape
        self.name = name

    def getResource(self, index):
        return self.D.iloc[index, :]

    def saveMetaData(self, filepath):
        metadata = dict(d=self.d, n=self.n, name=self.name)
        string = json.dumps(metadata, indent=2)
        with open(filepath, 'w') as f:
            f.write(string)
        return string

    def getMatrix(self):
        return self.D.as_matrix()


def convertDtype(l):
    try:
        return np.array(l, dtype="float")
    except:
        pass
    l = np.array(l, dtype=str)
    l[l == 'nan'] = 'NA'
    return l


class CSVDataSet(DataSet):
    """ A dataset living locally in a .csv file

    """

    def __init__(self,
                 csv_path,
                 index_column=None,
                 NA_val=".",
                 name="mydataset"):
        self.name = name

        # Load the data set
        D = pd.read_csv(csv_path, dtype="unicode")
        self.n, self.d = D.shape
        print("Dataset of size", self.n, "samples", self.d, "dimensions",
              "Loaded")

        # Convert to numeric all numeric rows
        D = D.replace(NA_val, "nan")
        print("Replacing all", NA_val, "with nan")
        d = []
        for c in D.columns:
            d.append(convertDtype(list(D[c])))
            print("Converting", c, end="\r\r")
        newcolumns = D.columns
        newindex = D.index
        D = list(d)
        D = pd.DataFrame(dict(zip(newcolumns, D)), index=newindex)

        # Set the index column as specified
        if index_column is not None:
            print("Setting index column as", index_column)
            D.index = D[index_column]
            print("Deleting", index_column, "from dataset")
            del D[index_column]

        self.D = D

        # Remove all columns which have all null values
        keep = []
        allnull = self.D.isnull().all(axis=0)
        for c in self.D.columns[allnull]:
            print("Removing column", c, "because it has all null values")
        keep = self.D.columns[~allnull]
        self.D = self.D[keep]

        # Remove all rows which have all null values
        allnull = self.D.isnull().all(axis=1)
        for r in self.D.index[allnull]:
            print("Removing row", r, "because it has all null values")
        keep = self.D.index[~allnull]
        self.D = self.D.loc[keep]
        n, d = self.D.shape
        print("Dataset of size", n, "samples", d, "dimensions", "Resulting")
        self.N = self.D.shape[0]

    def imputeColumns(self, numeric):
        keep = []
        keep = (self.D.dtypes == "float64").as_matrix()
        for c in self.D.columns[~keep]:
            print("Removing column", c, "because it is not numeric")
        self.D = self.D[self.D.columns[keep]]
        cmean = self.D.mean(axis=0)
        values = dict(list(zip(self.D.columns, cmean.as_matrix())))
        #self.D.fillna(value=values, inplace=True)
        d = self.D.as_matrix()
        for i, c in enumerate(self.D.columns):
            print("Imputing column", c, "with value", values[c])
            d[:, i][np.isnan(d[:, i])] = values[c]
        D = pd.DataFrame(d)
        D.index = self.D.index
        D.index.names = self.D.index.names
        D.columns = self.D.columns
        D.columns.names = self.D.columns.names
        self.D = D
        allzero = np.all(self.D.as_matrix() == 0, axis=0)
        for c in self.D.columns[allzero]:
            print("Removing column", c, "because it has all zero values")
        keep = self.D.columns[~allzero]
        allsame = np.std(self.D.as_matrix(), axis=0) == 0
        for c in self.D.columns[allsame]:
            print(
                "Removing column", c,
                "because it has all zero standard deviation (all values same)")
        keep = self.D.columns[~allsame]
        self.D = self.D[keep]
        n, d = self.D.shape
        print("Dataset of size", n, "samples", d, "dimensions", "Resulting")
        print("Dataset has", self.D.isnull().sum().sum(), "nans")
        print("Dataset has", np.sum(np.isinf(self.D.as_matrix())), "infs")

    def getResource(self, index):
        """Get a specific data point from the data set.

        Parameters
        ----------
        index : int or string
            The index of the data point in `D`, either positional or a string.

        Returns
        -------
        :obj:`ndarray`
            A ndarray of the data point.

        """
        if type(index) is int:
            return self.D.iloc[index].as_matrix()
        else:
            return self.D.loc[index].as_matrix()

    def getColumn(self, index):
        """Get a column of the dataframe.
 
        Parameters
        ----------
        index : int or string
            The index of the column in `D`, either positional or a string.

        Returns
        -------
        :obj:`ndarray`
            The values in the column.
        """
        if type(index) is int:
            return self.D.iloc[:, index].as_matrix()
        else:
            return self.D[index].as_matrix()

    def getColumnValues(self, index):
        """Get the unique values of a column.

        Parameters
        ----------
        index : int or string
            The index of the column in `D`, either positional or a string.

        Returns
        -------
        :obj:`ndarray`
            A ndarray of the unique values.

        """
        column = self.getColumn(index)
        if column.dtype == "float64":
            column = column[~np.isnan(column)]
        else:
            column = column[np.array([x != "NA" for x in column])]
        return np.unique(column)

    def getColumnDistribution(self, index):
        """Get the distribution of values in a column.

        Parameters
        ----------
        index : int or string
            The index of the column in `D`, either positional or a string.

        Returns
        -------
        :obj:`ndarray`, :obj:`ndarray`
            An array x of the unique labels, and an array y of the count of that label

        """
        x = self.getColumnValues(index)
        column = self.getColumn(index)
        y = [np.sum(column == v) for v in x]
        return x, y

    def getColumnNADist(self, index):
        column = self.getColumn(index)
        if column.dtype == "float64":
            na = np.sum([np.isnan(x) for x in column])
            not_na = len(column) - na
            return na, not_na
        else:
            na = np.sum([x == "NA" for x in column])
            not_na = len(column) - na
            return na, not_na
        return na, not_na

    def getColumnDescription(self, index, sep="\n"):
        """Get a description of the column.

        """
        desc = []
        if type(index) is int:
            index = self.D.columns.values[index]
        for i, name in enumerate(self.D.columns.names):
            desc.append(name + ": " + index[i])
        return sep.join(desc)

    def getLevelValues(self, index):
        return np.unique(self.D.columns.get_level_values(index))
