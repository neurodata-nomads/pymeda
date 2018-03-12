# PyMEDA
![](https://travis-ci.org/neurodata-nomads/pymeda.svg?branch=master)

PyMEDA is a python package for matrix exploratory data analysis (MEDA). It is inspired by the MEDA R package.

## Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#usage)

## Overview
**PyMEDA** is a data visualization package for understanding high dimensional data. It is powered by [Redlemur](https://github.com/neurodatadesign/lemur "Redlemur") and Plot.ly.

## System Requirements
  - **PyMEDA** was developed in Python 3.6. Currently, there is no plan to support Python 2.
  - Was developed and tested primarily on Mac OS (Sierra 10.12.6). It does not currently support Windows.
  - Requires no non-standard hardware to run.
  - Complete visualizations take roughly 2-3 minutes for data with >20 dimensions and >10<sup>6</sup> data points on a laptop (2.9GHz Intel i5, 8 GB RAM).

The following lists the dependencies for **PyMEDA**. Note that this not a comprehensive list of all the dependencies. Please check via `pip freeze` once **PyMEDA** is installed. 

```
jupyter==1.0.0,
numpy==1.13.1,
pandas==0.21.0,
scikit-learn==0.19.1,
plotly==2.2.3,
redlemur==0.10.0,
knor==0.0.1,
cython==0.27.3
```

## Installation Guide
**PyMEDA** can be installed either from `pip` or Github as shown below. 

### Install from pip

    pip install pymeda

### Install from Github

    git clone https://github.com/neurodata-nomads/pymeda
    cd pymeda
    python setup.py install

### Potential Installation Errors Due to Cython Dependency
#### 1. Xcode is out of date

    In file included from knor/cknor/libkcommon/clusters.cpp:23:
    knor/cknor/libkcommon/util.hpp:29:10: fatal error: ‘random’ file not found
    #include <random>
             ^
    1 error generated.
    error: command ‘/usr/bin/clang’ failed with exit status 1

#### Solution
Update your Xcode and Xcode command line tools to the latest version.

#### 2. Cython installation error

    from Cython.Build import cythonize
    ImportError: No module named Cython.Build

#### Solution
Install Cython via `pip install --upgrade cython`. This will install Cython, 
then install **PyMEDA** via one of the methods above.

#### 3. GCC compiler not installed

    knor/cknor/libkcommon/util.cpp:27:10: fatal error: numa.h: No such file or directory
     #include <numa.h>
              ^~~~~~~~
    compilation terminated.
    error: command 'gcc' failed with exit status 1

#### Solution
Install GCC compiler. Use `apt-get install build essential` or `yum install build-essential` depending
on your linux distribution.

## Usage
It is **_highly_** recommended that you use **PyMEDA** inside Jupyter notebook, which allows **PyMEDA** visualizations to be easily embedded. However, **PyMEDA** also supports embedding in static HTML pages. 

Please see [demo](https://github.com/neurodata-nomads/pymeda/blob/master/notebooks/Demo.ipynb "PyMEDA demo using iris dataset")
 here to view the usages using the iris dataset.
