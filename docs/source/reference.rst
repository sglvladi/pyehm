API Reference
=============

Core API
--------
The core components of PyEHM are the :class:`~.EHM` and :class:`~.EHM2` classes, that constitute implementations of the
EHM [EHM1]_ and EHM2 [EHM2]_ algorithms for data association.

The interfaces of these classes are documented below.

.. autoclass:: pyehm.core.EHM
    :members:

.. autoclass:: pyehm.core.EHM2
    :members:
    :show-inheritance:
    :inherited-members:

Utils API
---------
The :mod:`pyehm.utils` module contains helper classes and functions used by :mod:`pyehm.core`.

.. autoclass:: pyehm.utils.EHMNetNode
    :members:

.. autoclass:: pyehm.utils.EHM2NetNode
    :members:
    :show-inheritance:

.. autoclass:: pyehm.utils.EHMNet
    :members:

.. autoclass:: pyehm.utils.EHM2Tree
    :members:

.. autoclass:: pyehm.utils.Cluster
    :members:

.. autofunction:: pyehm.utils.gen_clusters


Plugins
-------

Stone Soup
^^^^^^^^^^

.. autoclass:: pyehm.plugins.stonesoup.JPDAWithEHM
    :members: associate
    :show-inheritance:

.. autoclass:: pyehm.plugins.stonesoup.JPDAWithEHM2
    :members: associate
    :show-inheritance:

