# -*- coding: utf-8 -*-

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("pyehm").version
except DistributionNotFound:
    # package is not installed
    pass
