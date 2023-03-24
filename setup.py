from setuptools import setup, find_packages
from glob import glob
import platform

# from pyehm import __version__

from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "2.0a1"

with open('README.md') as f:
    long_description = f.read()

cpp_args = []
# if platform.system() == 'Windows':
#     cpp_args = ['/std:c++20']
# else:
#     cpp_args = ['-std=c++20']

core_sources = sorted(glob("./src/core/*.cpp"))
net_sources = sorted(glob("./src/net/*.cpp"))
utils_sources = sorted(glob("./src/utils/*.cpp"))

ext_module = Pybind11Extension(
    '_pyehm',
    sources=['src/module.cpp', 'src/Docstrings.cpp', *core_sources, *net_sources, *utils_sources],
    include_dirs=[r'./src', r'./include'],
    language='c++',
    extra_compile_args=cpp_args,
    define_macros=[('VERSION_INFO', __version__)],
)

setup(
    name='pyehm',
    version=__version__,
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
    install_requires=['numpy', 'networkx', 'stonesoup', 'setuptools>=42', 'pydot', 'matplotlib'],
    extras_require={
        'dev': ['pytest-flake8', 'pytest-cov', 'flake8<5', 'sphinx', 'sphinx_rtd_theme',
                'sphinx-gallery>=0.8']
    },
    entry_points={'stonesoup.plugins': 'pyehm = pyehm.plugins.stonesoup'},
    python_requires='>=3.7',
    keywords=['python', 'pyehm', 'ehm'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    cmdclass={"build_ext": build_ext},
    ext_modules=[ext_module],
)
