#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

from pyehm import __version__ as version

with open('README.md') as f:
    long_description = f.read()

# Setting up
setup(
    name="pyehm",
    version=version,
    author="Lyudmil Vladimirov",
    author_email="sglvladi@liverpool.ac.uk",
    maintainer="University of Liverpool",
    url='https://github.com/sglvladi/pyehm',
    description='Python Efficient Hypothesis Management (PyEHM)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    project_urls={
        'Documentation': 'https://pyehm.rtfd.io/',
        'Source': 'https://github.com/sglvladi/pyehm',
        'Issue Tracker': 'https://github.com/sglvladi/pyehm/issues'
    },
    packages=find_packages(exclude=('docs', '*.tests')),
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    install_requires=['numpy', 'networkx', 'stonesoup', 'setuptools>=42', 'pydot', 'matplotlib'],
    extras_require={
        'dev': ['pytest-flake8', 'pytest-cov', 'flake8<5', 'sphinx', 'sphinx_rtd_theme',
                'sphinx-gallery>=0.8']
    },
    entry_points={'stonesoup.plugins': 'pyehm = pyehm.plugins.stonesoup'},
    python_requires='>=3.7',
    keywords=['python', 'pyehm', 'ehm'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ]
)
