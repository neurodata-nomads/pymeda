# PyMEDA
[![](https://img.shields.io/pypi/v/pymeda/svg)](https://pypi.python.org/pypi/pymeda)
![](https://travis-ci.org/neurodata-nomads/pymeda.svg?branch=master)

PyMEDA is a python package for matrix exploratory data analysis (MEDA). 

## System Requirements
The software has been tested macOS Sierra 10.12.6 (2.9GHz Intel Core i5).

### Python depedencies:
jupyter==1.0.0<br/>
nose2==0.6.5<br/>
numpy==1.13.1<br/>
pandas==0.21.0<br/>
plotly==2.2.3<br/>
scikit-image==0.13.1<br/>
scipy==1.0.0<br/>
-e git://github.com/j1c/lemur.git@clustering#egg=redlemur<br/>
scikit-learn==0.19.1<br/>

## Installation Guide
This assumes you have these tools already installed:

1. Python 3.6 ([Download](https://www.python.org/downloads/))
2. Git ([Installation](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))
2. Virtualenv ([Installation and usage](https://help.dreamhost.com/hc/en-us/articles/115000695551-Installing-and-using-Python-s-virtualenv-using-Python-3))
3. Virtualenvwrapper ([Installation](http://virtualenvwrapper.readthedocs.io/en/latest/install.html))

If you do not have these tools installed, you can click the links to download and install them. If you do have these tools installed, you can go ahead with the following. It should take about 2-3 minutes.
1. Clone this repository
    ```
    git clone https://github.com/neurodata-nomads/pymeda.git
    ```
2. Navigate to the repository directory
    ```
    cd pymeda
    ```
3. Create a virtualenvironment called synaptome
    ```
    mkvirtualenv -p python3 pymeda
    ```
2. Install the dependencies using pip and requirements.txt
    ```
    pip install -r requirements.txt
    ```
3. Make the synaptome virtual environment available in jupyter notebook.
    ```
    python -m ipykernel install --user --name=pymeda
    ```
4. Run jupyter notebook
    ```
    jupyter notebook
    ```
