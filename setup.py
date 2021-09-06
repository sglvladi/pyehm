from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Efficient Hypothesis Management (EHM) Python Implementation'

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

# Setting up
setup(
    name="ehm",
    version=VERSION,
    author="Lyudmil Vladimirov",
    author_email="sglvladi@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    install_requires=['numpy', 'networkx', 'stonesoup', 'setuptools>=42', 'pydot'],
    extras_require={
        'dev': ['pytest-flake8', 'pytest-cov']
    },
    entry_points={'stonesoup.plugins': 'ehm = ehm'},
    python_requires='>=3.6',
    keywords=['python', 'ehm'],
    classifiers=[
        "Development Status :: 1 - Beta",
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)"
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ]
)