from typing import Union

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

from .net import EHMNet, EHM2Net, EHMTree
from .utils import to_nx_graph


def plot_net(net: Union[EHMNet, EHM2Net], ax: plt.Axes = None, annotate=True):
    """Plot the net.

    Parameters
    ----------
    net : :class:`~.EHMNet` | :class:`~.EHM2Net`
        The net to plot.
    ax : :class:`matplotlib.axes.Axes`
        Axes on which to plot the net. If ``None``, a new figure and axes will be created.
    annotate : :class:`bool`
        Flag that dictates whether to draw node and edge labels on the plotted net. The default is ``True``
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    g = to_nx_graph(net)
    pos = graphviz_layout(g, prog="dot")
    nx.draw(g, pos, ax=ax, node_size=0)
    if annotate:
        labels = dict()
        for n in g.nodes:
            t = g.nodes[n]['track']
            s = str(g.nodes[n]['identity']) if len(g.nodes[n]['identity']) else 'Ø'
            if t is not None:
                labels[n] = '{{{}, {}}}'.format(t, s)
            else:
                labels[n] = 'Ø'
        pos_labels = {}
        for node, coords in pos.items():
            pos_labels[node] = (coords[0] + 10, coords[1])
        nx.draw_networkx_labels(g, pos_labels, ax=ax, labels=labels, horizontalalignment='left')
        edge_labels = nx.get_edge_attributes(g, 'detections')
        for key in edge_labels:
            edge_labels[key] = str(edge_labels[key]).replace('{', '').replace('}', '')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)


def plot_tree(tree: EHMTree, ax: plt.Axes = None, annotate=True):
    """Plot the tree.

    Parameters
    ----------
    tree : :class:`~.EHMTree`
        The tree to plot.
    ax: :class:`matplotlib.axes.Axes`
        Axes on which to plot the tree. If ``None``, a new figure and axes will be created.
    annotate : :class:`bool`
        Flag that dictates whether to draw node labels on the plotted tree. The default is ``True``
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    g = to_nx_graph(tree)
    pos = graphviz_layout(g, prog="dot")
    nx.draw(g, pos, ax=ax)
    labels = {n: g.nodes[n]['track'] for n in g.nodes}  # if g.nodes[n]['leaf']}
    pos_labels = {}
    if annotate:
        for node, coords in pos.items():
            # if g.nodes[node]['leaf']:
            pos_labels[node] = (coords[0], coords[1])
        nx.draw_networkx_labels(g, pos_labels, ax=ax, labels=labels, font_color='white')
