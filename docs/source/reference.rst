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

Net API
-------
The :mod:`pyehm.net` module contains classes that implement the structures (nets, nodes, trees) constructed by the
:class:`~.EHM` and :class:`~.EHM2` classes.

.. autoclass:: pyehm.net.EHMNetNode
    :members:

.. autoclass:: pyehm.net.EHM2NetNode
    :members:

.. autoclass:: pyehm.net.EHMNet
    :members:

.. autoclass:: pyehm.net.EHM2Net
    :members:

.. autoclass:: pyehm.net.EHM2Tree
    :members:

Utils API
---------
The :mod:`pyehm.utils` module contains helper classes and functions.

.. autoclass:: pyehm.utils.Cluster
    :members:

.. autofunction:: pyehm.utils.gen_clusters

.. autofunction:: pyehm.utils.to_nx_graph

Plotting API
------------
The :mod:`pyehm.plot` module contains helper functions for plotting the nets and trees constructed by the
:class:`~.EHM` and :class:`~.EHM2` classes.

.. warning::
    The plotting functions require `Graphviz <https://graphviz.org/>`_ to be installed and on the ``PATH``.

.. autofunction:: pyehm.plotting.plot_net

.. autofunction:: pyehm.plotting.plot_tree



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

