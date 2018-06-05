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

VERSION = '0.1.10'

# Install Cython
try:
    import Cython
except ImportError:
    import pip
    # For installing Cython due to pip.main removal in Pip10
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Cython'])

    import Cython

setup(
    name='pymeda',
    version=VERSION,
    description='Matrix Exploratory Data Analysis',
    url='https://github.com/neurodata-nomads/pymeda',
    download_url='https://github.com/neurodata-nomads/pymeda/tarball/0.1.5',
    author='Jaewon Chung',
    author_email='j1c@jhu.edu',
    license='MIT',
    keywords='data visualization analysis',
    packages=['pymeda'],  # Required
    setup_requires=['Cython'],
    install_requires=[
        'Cython',
        'jupyter==1.0.0',
        'redlemur',
        'knor==0.0.1',
    ],
    #dependency_links=['https://github.com/j1c/lemur#egg=redlemur'],
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
