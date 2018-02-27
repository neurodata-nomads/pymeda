"""A setuptools based setup module.
See:
https://packaging.python.org/tutorials/distributing-packages/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

VERSION = '0.1.1'

setup(
    name='pymeda',
    version=VERSION,
    description='Matrix Exploratory Data Analysis',
    url='https://github.com/neurodata-nomads/pymeda',
    download_url='https://github.com/neurodata-nomads/pymeda/tarball/0.0.4',
    author='Jaewon Chung',
    author_email='j1c@jhu.edu',
    license='MIT',
    keywords='data visualization analysis',
    packages=['pymeda'],  # Required
    install_requires=[
        'colorlover==0.2.1', 'jupyter==1.0.0', 'numpy==1.13.1',
        'pandas==0.21.0', 'plotly==2.2.3', 'scipy==1.0.0', 'redlemur==0.10.0',
        'scikit-learn==0.19.1', 'cython', 'knor', 'boto3==1.4.7',
        'nilearn==0.4.0', 'matplotlib==2.1.0', 'nose', 'imageio'
    ],
    package_data={
        'pymeda': ['*.html'],
    },
    include_package_data=True,
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ])
