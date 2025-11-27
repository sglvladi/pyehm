# -*- coding: utf-8 -*-

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyehm")
except PackageNotFoudError:
    # package is not installed
    pass
