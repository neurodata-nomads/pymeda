# PyMEDA
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
redlemur,
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

## Docker
**PyMEDA** is available through Dockerhub, and can be run directly with the following container (and any additional options you may require for Docker, such as volume mounting):

    docker run -p 8888:8888 nomads/pymeda:latest

You should see message that is similar to below:

    Executing the command: jupyter notebook
    [I 15:33:00.567 NotebookApp] Writing notebook server cookie secret to /home/jovyan/.local/share/jupyter/runtime/notebook_cookie_secret
    [W 15:33:01.084 NotebookApp] WARNING: The notebook server is listening on all IP addresses and not using encryption. This is not recommended.
    [I 15:33:01.150 NotebookApp] JupyterLab alpha preview extension loaded from /opt/conda/lib/python3.6/site-packages/jupyterlab
    [I 15:33:01.150 NotebookApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
    [I 15:33:01.155 NotebookApp] Serving notebooks from local directory: /home/jovyan
    [I 15:33:01.156 NotebookApp] 0 active kernels
    [I 15:33:01.156 NotebookApp] The Jupyter Notebook is running at:
    [I 15:33:01.157 NotebookApp] http://[all ip addresses on your system]:8888/?token=112bb073331f1460b73768c76dffb2f87ac1d4ca7870d46a
    [I 15:33:01.157 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
    [C 15:33:01.160 NotebookApp]

        Copy/paste this URL into your browser when you connect for the first time,
        to login with a token:
            http://localhost:8888/?token=112bb073331f1460b73768c76dffb2f87ac1d4ca7870d46a

Copy and paste the URL into your browser, and you can now use **PyMEDA** in Jupyter.

## Potential Installation Errors Due to Cython Dependency
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

If you are using Docker, then there should be a folder called `work` with the Demo already downloaded for you.
