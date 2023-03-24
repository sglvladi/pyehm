from typing import Union

import networkx as nx

from _pyehm.utils import *
from .net import EHMNet, EHM2Net, EHMNetNode, EHM2NetNode, EHM2Tree


def to_nx_graph(obj: Union[EHMNet, EHM2Net, EHM2Tree]) -> nx.Graph:
    """Get a NetworkX representation of a net or tree. Mainly used for plotting.

    Parameters
    ----------
    obj : :class:`~.EHMNet` | :class:`~.EHM2Net` | :class:`~.EHM2Tree`
        The object to convert to a NetworkX graph.

    Returns
    -------
    :class:`networkx.Graph`
        The NetworkX graph representation of the object.

    """
    if isinstance(obj, EHMNet):
        g = nx.Graph()
        for child in sorted(obj.nodes, key=lambda x: x.layer):
            parents = obj.get_parents(child)
            track = child.layer + 1 if child.layer + 2 < obj.num_layers else None
            identity = child.identity
            g.add_node(child.id, track=track, identity=identity)
            for parent in parents:
                label = obj.get_edges(parent, child)
                g.add_edge(parent.id, child.id, detections=label)
        return g
    elif isinstance(obj, EHM2Net):
        g = nx.Graph()
        for parent in obj.nodes:
            track = parent.track if parent.track != -1 else None
            identity = parent.identity
            g.add_node(parent.id, track=track, identity=identity)
            for detection in range(obj.validation_matrix.shape[1]):
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