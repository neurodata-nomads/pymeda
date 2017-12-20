# Synaptome
Running MEDA on Synapses

## System Requirements
The software has been tested macOS Sierra 10.12.6 (2.9GHz Intel Core i5).

### Python depedencies:
blosc==1.4.4<br/>
colorlover==0.2.1<br/>
-e git://github.com/j1c/intern.git#egg=intern<br/>
jupyter==1.0.0<br/>
mock==2.0.0<br/>
nose2==0.6.5<br/>
numpy==1.13.1<br/>
pandas==0.21.0<br/>
pbr==3.1.1<br/>
plotly==2.2.3<br/>
requests==2.11.1<br/>
scikit-image==0.13.1<br/>
scipy==1.0.0<br/>
six==1.10.0<br/>
-e git://github.com/j1c/lemur.git@j1c-dev#egg=redlemur<br/>
scikit-learn==0.19.1<br/>
seaborn==0.8.1<br/>

## Installation Guide
This assumes you have these tools already installed:

1. Python 3.6 ([Download](https://www.python.org/downloads/))
2. Git ([Installation](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))
2. Virtualenv ([Installation and usage](https://help.dreamhost.com/hc/en-us/articles/115000695551-Installing-and-using-Python-s-virtualenv-using-Python-3))
3. Virtualenvwrapper ([Installation](http://virtualenvwrapper.readthedocs.io/en/latest/install.html))

If you do not have these tools installed, you can click the links to download and install them. If you do have these tools installed, you can go ahead with the following. It should take about 2-3 minutes.
1. Clone this repository
    ```
    git clone https://github.com/j1c/synaptome.git
    ```
2. Navigate to the repository directory
    ```
    cd synaptome
    ```
3. Create a virtualenvironment called synaptome
    ```
    mkvirtualenv -p python3 synaptome
    ```
2. Install the dependencies using pip and requirements.txt
    ```
    pip install -r requirements.txt
    ```
3. Make the synaptome virtual environment available in jupyter notebook.
    ```
    python -m ipykernel install --user --name=synaptome
    ```
4. Run jupyter notebook
    ```
    jupyter notebook
    ```

## Demo
Open the `api_key.txt` text file in the repository, and copy/paste your token into the file. This is required for grabbing data from BOSS, and later, uploading data back to the BOSS.

1. Navigate to the jupyter tab in your browser. If you closed it by accident, you can reopen it using the following url:
    ```
    localhost:8888/tree
    ```
2. Open the `demo.ipynb`
3. You can run each cell to view the results.
