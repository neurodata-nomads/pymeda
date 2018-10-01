"""A setuptools based setup module.
See:
https://packaging.python.org/tutorials/distributing-packages/
https://github.com/pypa/sampleproject
"""

import sys

# Always prefer setuptools over distutils
from setuptools import setup

VERSION = '0.1.13.2'

# Install Cython
try:
    import Cython
except ImportError:
    #import pip
    # For installing Cython due to pip.main removal in Pip10
    import subprocess
    errno = subprocess.call([sys.executable, '-m', 'pip', 'install', 'cython'])
    if errno:
        print("Please install Cython.")
        raise SystemExit(errno)
    else:
        import Cython
        from Cython.Build import cythonize
        from Cython.Distutils import build_ext

setup(
    name='pymeda',
    version=VERSION,
    description='Matrix Exploratory Data Analysis',
    url='https://github.com/neurodata-nomads/pymeda',
    author='Jaewon Chung',
    author_email='j1c@jhu.edu',
    license='MIT',
    keywords='data visualization analysis',
    packages=['pymeda'],  # Required
    setup_requires=['Cython'],
    install_requires=[
        'Cython',
        'jinja2',
        'ipykernel',
        'pandas',
        'numpy',
        'scipy',
        'plotly==2.2.2',
        'matplotlib',
        'colorlover',
        'scikit-learn',
        'knor==0.0.2',
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
