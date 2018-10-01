FROM jupyter/minimal-notebook

USER root

RUN apt-get update && apt-get install -y libnuma-dev libnuma1 build-essential
#RUN apt-get install -y python3-dev python3-pip
RUN pip install pymeda
#RUN pip3 install jupyter
#RUN pip3 install 'ipython<7.0'

USER $NB_UID

RUN cd work/ && wget https://raw.githubusercontent.com/neurodata-nomads/pymeda/master/notebooks/Demo.ipynb