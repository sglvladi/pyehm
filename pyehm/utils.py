from typing import Union

import networkx as nx

from _pyehm.utils import Cluster, gen_clusters as _gen_clusters  # noqa: F401
from ._compat import bind, is_validation_matrix, remember_likelihood_matrices, to_log_likelihoods, warn_deprecated
from .net import EHMNet, EHMTree


def gen_clusters(*args, **kwargs):
    legacy = (len(args) == 2 or 'validation_matrix' in kwargs or 'likelihood_matrix' in kwargs
              or (len(args) == 1 and not kwargs and is_validation_matrix(args[0])))
    if legacy:
        validation_matrix, likelihood_matrix = bind(args, kwargs, ('validation_matrix', 'likelihood_matrix'),
                                                    defaults={'likelihood_matrix': None})
        warn_deprecated('Passing a validation matrix to gen_clusters is deprecated; pass a log-likelihood matrix '
                        'instead, where -inf denotes invalid associations.')
        clusters = _gen_clusters(to_log_likelihoods(validation_matrix, likelihood_matrix))
        return remember_likelihood_matrices(clusters, likelihood_matrix)
    return _gen_clusters(*args, **kwargs)


gen_clusters.__doc__ = _gen_clusters.__doc__.rstrip('\n') + '''

gen_clusters(validation_matrix: numpy.ndarray, likelihood_matrix: numpy.ndarray = numpy.array([])) -> List[Cluster]
Cluster tracks into groups sharing detections, given a validation matrix as in pyehm 2.2

.. deprecated:: 2.3
    Will be removed in pyehm 3.0. Pass a log-likelihood matrix instead, where ``-numpy.inf`` denotes invalid
    associations. A matrix that contains only 0s and 1s is interpreted as a validation matrix.

Parameters
----------
validation_matrix: :class:`numpy.ndarray`
    An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
    (aka. valid) associations between tracks and detections. The first column corresponds
    to the null hypothesis (hence contains all ones).
likelihood_matrix: :class:`numpy.ndarray`
    A matrix of shape (num_tracks, num_detections + 1) containing the unnormalised
    likelihoods for all combinations of tracks and detections. The first column corresponds
    to the null hypothesis. Defaults to an empty array.

Returns
-------
:class:`list` of :class:`~.Cluster`
    The clusters of tracks.
'''


def to_nx_graph(obj: Union[EHMNet, EHMTree]) -> nx.Graph:
    """Get a NetworkX representation of a net or tree. Mainly used for plotting.

    Parameters
    ----------
    obj : :class:`~.EHMNet` | :class:`~.EHM2Net` | :class:`~.EHMTree`
        The object to convert to a NetworkX graph.

    Returns
    -------
    :class:`networkx.Graph`
        The NetworkX graph representation of the object.

    """
    if isinstance(obj, EHMNet):
        g = nx.Graph()
        for parent in obj.nodes:
            track = parent.track if parent.track != -1 else None
            identity = parent.identity
            g.add_node(parent.id, track=track, identity=identity)
            for detection in range(obj.log_likelihood_matrix.shape[1]):
                children = obj.get_children_per_detection(parent, detection)
                for child in children:
                    if not g.has_node(child.id):
                        track = child.track
                        identity = child.identity
                        g.add_node(child.id, track=track, identity=identity)
                    if not g.has_edge(parent.id, child.id):
                        g.add_edge(parent.id, child.id, detections={detection})
                    else:
                        g[parent.id][child.id]["detections"].add(detection)
        return g
    else:
        g = nx.Graph()
        return _traverse_tree_nx(obj, g)


def _traverse_tree_nx(tree, g, parent=None):
    child = g.number_of_nodes() + 1
    track = tree.track
    detections = tree.detections
    g.add_node(child, track=track, detections=detections)
    if parent:
        g.add_edge(parent, child)
    for sub_tree in tree.children:
        _traverse_tree_nx(sub_tree, g, child)
    return g
