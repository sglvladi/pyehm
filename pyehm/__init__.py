# -*- coding: utf-8 -*-

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyehm")
except PackageNotFoundError:
    # package is not installed
    pass
