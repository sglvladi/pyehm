# -*- coding: utf-8 -*-
from typing import Union, List, Sequence

import networkx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.algorithms.components.connected import connected_components


class EHMNetNode:
    """A node in the :class:`~.EHMNet` constructed by :class:`~.EHM`.

    Parameters
    ----------
    layer: :class:`int`
        Index of the network layer in which the node is placed. Since a different layer in the network is built for
        each track, this also represented the index of the track this node relates to.
    identity: :class:`set` of :class:`int`
        The identity of the node. As per Section 3.1 of [EHM1]_, "the identity for each node is an indication of how
        measurement assignments made for tracks already considered affect assignments for tracks remaining to be
        considered".
    """
    def __init__(self, layer, identity=None):
        # Index of the layer (track) in the network
        self.layer = layer
        # Identity of the node
        self.identity = identity if identity else set()
        # Index of the node when added to the network. This is set by the network and
        # should not be edited.
        self.ind = None

    def __repr__(self):
        return 'EHMNetNode(ind={}, layer={}, identity={})'.format(self.ind, self.layer, self.identity)


class EHM2NetNode(EHMNetNode):
    """A node in the :class:`~.EHMNet` constructed by :class:`~.EHM2`.

    Parameters
    ----------
    layer: :class:`int`
        Index of the network layer in which the node is placed.
    track: :class:`int`
        Index of track this node relates to.
    subnet: :class:`int`
        Index of subnet to which the node belongs.
    identity: :class:`set` of :class:`int`
        The identity of the node. As per Section 3.1 of [EHM1]_, "the identity for each node is an indication of how
        measurement assignments made for tracks already considered affect assignments for tracks remaining to be
        considered".
    """
    def __init__(self, layer, track=None, subnet=0, identity=None):
        super().__init__(layer, identity)
        # Index of track this node relates to
        self.track = track
        # Index of subnet the node belongs to
        self.subnet = subnet

    def __repr__(self):
        return 'EHM2NetNode(ind={}, layer={}, track={}, subnet={}, identity={})'.format(self.ind, self.layer,
                                                                                        self.track, self.subnet,
                                                                                        self.identity)


class EHMNet:
    """Represents the nets constructed by :class:`~.EHM` and :class:`~.EHM2`.

    Parameters
    ----------
    nodes: :class:`list` of :class:`~.EHMNetNode` or :class:`~.EHM2NetNode`
        The nodes comprising the net.
    validation_matrix: :class:`numpy.ndarray`
        An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
        (aka. valid) associations between tracks and detections. The first column corresponds
        to the null hypothesis (hence contains all ones).
    edges: :class:`dict`
        A dictionary that represents the edges between nodes in the network. The dictionary keys are tuples of the form
        ```(parent, child)```, where ```parent``` and ```child``` are the source and target nodes respectively. The
        values of the dictionary are the measurement indices that describe the parent-child relationship.
    """
    def __init__(self, nodes, validation_matrix, edges=None):

        self._num_layers = 0

        for n_i, node in enumerate(nodes):
            node.ind = n_i
            if isinstance(node, EHM2NetNode):
                if node.layer + 1 > self._num_layers:
                    self._num_layers = node.layer + 1
            else:
                if node.layer + 2 > self._num_layers:
                    self._num_layers = node.layer + 2
        self._nodes = nodes
        self.validation_matrix = validation_matrix
        self.edges = edges if edges is not None else dict()
        self.parents_per_detection = dict()
        self.children_per_detection = dict()
        self.nodes_per_track = dict()

        self._parents = dict()
        self._children = dict()

    @property
    def root(self) -> Union[EHMNetNode, EHM2NetNode]:
        """The root node of the net."""
        return self.nodes[0]

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the net"""
        return len(self._nodes)

    @property
    def num_layers(self) -> int:
        """Number of layers in the net"""
        return self._num_layers

    @property
    def nodes(self) -> Union[List[EHMNetNode], List[EHM2NetNode]]:
        """The nodes comprising the net"""
        return self._nodes

    @property
    def nodes_forward(self) -> Union[Sequence[EHMNetNode], Sequence[EHM2NetNode]]:
        """The net nodes, ordered by increasing layer"""
        return sorted(self.nodes, key=lambda x: x.layer)

    @property
    def nx_graph(self) -> networkx.Graph:
        """A NetworkX representation of the net. Mainly used for plotting the net."""
        g = nx.Graph()
        for child in sorted(self.nodes, key=lambda x: x.layer):
            parents = self.get_parents(child)
            if isinstance(child, EHM2NetNode):
                track = child.track
            else:
                track = child.layer + 1 if child.layer + 2 < self.num_layers else None
            identity = child.identity
            g.add_node(child.ind, track=track, identity=identity)
            for parent in parents:
                label = str(self.edges[(parent, child)]).replace('{', '').replace('}', '')
                g.add_edge(parent.ind, child.ind, detections=label)
        return g

    def add_node(self, node: Union[EHMNetNode, EHM2NetNode], parent: Union[EHMNetNode, EHM2NetNode], detection: int):
        """Add a new node in the network.

        Parameters
        ----------
        node: :class:`~.EHMNetNode` or :class:`~.EHM2NetNode`
            The node to be added.
        parent: :class:`~.EHMNetNode` or :class:`~.EHM2NetNode`
            The parent of the node.
        detection: :class:`int`
            Index of measurement representing the parent child relationship.
        """
        # Set the node index
        node.ind = len(self.nodes)
        # Add node to graph
        self.nodes.append(node)
        # Create edge from parent to child
        self.edges[(parent, node)] = {detection}
        # Create parent-child-detection look-up
        self.parents_per_detection[(node, detection)] = {parent}
        self._parents[node] = {parent}
        try:
            self.children_per_detection[(parent, detection)].add(node)
        except KeyError:
            self.children_per_detection[(parent, detection)] = {node}
        try:
            self._children[parent].add(node)
        except KeyError:
            self._children[parent] = {node}

        if isinstance(node, EHM2NetNode):
            if node.layer + 1 > self._num_layers:
                self._num_layers = node.layer + 1
        else:
            if node.layer + 2 > self._num_layers:
                self._num_layers = node.layer + 2

    def add_edge(self, parent: Union[EHMNetNode, EHM2NetNode], child: Union[EHMNetNode, EHM2NetNode], detection: int):
        """ Add edge between two nodes, or update an already existing edge by adding the detection to it.

        Parameters
        ----------
        parent: :class:`~.EHMNetNode` or :class:`~.EHM2NetNode`
            The parent node, i.e. the source of the edge.
        child: :class:`~.EHMNetNode` or :class:`~.EHM2NetNode`
            The child node, i.e. the target of the edge.
        detection: :class:`int`
            Index of measurement representing the parent child relationship.
        """
        try:
            self.edges[(parent, child)].add(detection)
        except KeyError:
            self.edges[(parent, child)] = {detection}
        try:
            self.parents_per_detection[(child, detection)].add(parent)
        except KeyError:
            self.parents_per_detection[(child, detection)] = {parent}
        try:
            self.children_per_detection[(parent, detection)].add(child)
        except KeyError:
            self.children_per_detection[(parent, detection)] = {child}
        try:
            self._children[parent].add(child)
        except KeyError:
            self._children[parent] = {child}
        try:
            self._parents[child].add(parent)
        except KeyError:
            self._parents[child] = {parent}

    def get_parents(self, node: Union[EHMNetNode, EHM2NetNode]) -> Union[Sequence[EHMNetNode], Sequence[EHM2NetNode]]:
        """Get the parents of a node.

        Parameters
        ----------
        node: :class:`~.EHMNetNode` or :class:`~.EHM2NetNode`
            The node whose parents should be returned

        Returns
        -------
        :class:`list` of :class:`~.EHMNetNode` or :class:`~.EHM2NetNode`
            List of parent nodes
        """
        try:
            parents = list(self._parents[node])
        except KeyError:
            parents = []
        return parents  # [edge[0] for edge in self.edges if edge[1] == node]

    def get_children(self, node: Union[EHMNetNode, EHM2NetNode]) -> Union[Sequence[EHMNetNode], Sequence[EHM2NetNode]]:
        """Get the children of a node.

        Parameters
        ----------
        node: :class:`~.EHMNetNode` or :class:`~.EHM2NetNode`
            The node whose children should be returned

        Returns
        -------
        :class:`list` of :class:`~.EHMNetNode` or :class:`~.EHM2NetNode`
            List of child nodes
        """
        try:
            children = list(self._children[node])
        except KeyError:
            children = []
        return children  # [edge[1] for edge in self.edges if edge[0] == node]

    def plot(self, ax: plt.Axes = None, annotate=True):
        """Plot the net.

        Parameters
        ----------
        ax: :class:`matplotlib.axes.Axes`
            Axis on which to plot the net
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()
        g = self.nx_graph
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
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)


class EHM2Tree:
    """Represents the track tree structure generated by :func:`~pyehm.core.EHM2.construct_tree`.

    The :class:`~.EHM2Tree` object represents both a tree as well as the root node in the tree.

    Parameters
    ----------
    track: :class:`int`
        The index of the track represented by the root node of the tree
    children: :class:`list` of :class:`~.EHM2Tree`
        Sub-trees that are children of the current tree
    detections: :class:`set` of :class:`int`
        Set of accumulated detections
    subtree: :class:`int`
        Index of subtree the current tree belongs to.
    """
    def __init__(self, track, children, detections, subtree):
        self.track = track
        self.children = children
        self.detections = detections
        self.subtree = subtree

    @property
    def depth(self) -> int:
        """The depth of the tree"""
        depth = 1
        c_depth = 0
        for child in self.children:
            child_depth = child.depth
            if child_depth > c_depth:
                c_depth = child_depth
        return depth + c_depth

    @property
    def nodes(self) -> List['EHM2Tree']:
        """The nodes/subtrees in the tree"""
        nodes = [self]
        for child in self.children:
            nodes += child.nodes
        return nodes

    @property
    def nx_graph(self) -> networkx.Graph:
        """A NetworkX representation of the tree. Mainly used for plotting the tree."""
        g = nx.Graph()
        return self._traverse_tree_nx(self, g)

    @classmethod
    def _traverse_tree_nx(cls, tree, g, parent=None):
        child = g.number_of_nodes() + 1
        track = tree.track
        detections = tree.detections
        g.add_node(child, track=track, detections=detections)
        if parent:
            g.add_edge(parent, child)
        for sub_tree in tree.children:
            cls._traverse_tree_nx(sub_tree, g, child)
        return g

    def plot(self, ax: plt.Axes = None):
        """Plot the tree.

        Parameters
        ----------
        ax: :class:`matplotlib.axes.Axes`
            Axis on which to plot the tree
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()
        g = self.nx_graph
        pos = graphviz_layout(g, prog="dot")
        nx.draw(g, pos, ax=ax)
        labels = {n: g.nodes[n]['track'] for n in g.nodes}  # if g.nodes[n]['leaf']}
        pos_labels = {}
        for node, coords in pos.items():
            # if g.nodes[node]['leaf']:
            pos_labels[node] = (coords[0], coords[1])
        nx.draw_networkx_labels(g, pos_labels, ax=ax, labels=labels, font_color='white')


class Cluster:
    """A cluster of tracks sharing common detections.

    Parameters
    ----------
    tracks: :class:`list` of `int`
        Indices of tracks in cluster
    detections: :class:`list` of `int`
        Indices of detections in cluster
    validation_matrix: :class:`numpy.ndarray`
        The validation matrix for tracks and detections in the cluster
    likelihood_matrix: :class:`numpy.ndarray`
        The likelihood matrix for tracks and detections in the cluster

    """
    def __init__(self, tracks=None, detections=None, validation_matrix=None, likelihood_matrix=None):
        self.tracks = tracks
        self.detections = detections
        self.validation_matrix = validation_matrix
        self.likelihood_matrix = likelihood_matrix


def gen_clusters(validation_matrix, likelihood_matrix=None):
    """Cluster tracks into groups sharing detections

    Parameters
    ----------
    validation_matrix: :class:`numpy.ndarray`
        An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
        (aka. valid) associations between tracks and detections. The first column corresponds
        to the null hypothesis (hence contains all ones).
    likelihood_matrix: :class:`numpy.ndarray`
        A matrix of shape (num_tracks, num_detections + 1) containing the unnormalised
        likelihoods for all combinations of tracks and detections. The first column corresponds
        to the null hypothesis. The default is None, in which case the likelihood matrices of
        the generated clusters will also be None.

    Returns
    -------
    list of :class:`Cluster` objects
        A list of :class:`Cluster` objects, where each cluster contains the indices of the rows
        (tracks) and columns (detections) pertaining to the cluster
    list of int
        A list of row (track) indices that have not been associated to any detections
    """

    # Validation matrix for all detections except null
    validation_matrix_true = validation_matrix[:, 1:]

    # Initiate parameters
    num_tracks, num_detections = np.shape(validation_matrix_true)  # Number of tracks

    # Form clusters of tracks sharing measurements
    missed_tracks = set([i for i in range(num_tracks)])
    clusters = list()

    # List of tracks gated for each detection
    v_lists = [np.flatnonzero(validation_matrix_true[:, detection]) for detection in range(num_detections)]

    # Get clusters of tracks sharing common detections
    G = to_graph(v_lists)
    track_clusters = [t for t in connected_components(G)]

    # Create cluster objects that contain the indices of tracks (rows) and detections (cols)
    for tracks in track_clusters:
        v_detections = {0}
        for track in tracks:
            v_detections |= set(np.flatnonzero(validation_matrix_true[track, :]) + 1)
        # Extract validation and likelihood matrices for cluster
        tracks = sorted(tracks)
        v_detections = sorted(v_detections)
        c_validation_matrix = validation_matrix[tracks, :][:, v_detections]
        if likelihood_matrix is not None:
            c_likelihood_matrix = likelihood_matrix[tracks, :][:, v_detections]
        else:
            c_likelihood_matrix = None
        clusters.append(Cluster(tracks, v_detections, c_validation_matrix, c_likelihood_matrix))

    # Get tracks (rows) that are not associated to any detections
    detected_tracks = set([j for i in track_clusters for j in i])
    missed_tracks = missed_tracks - detected_tracks

    return clusters, list(missed_tracks)


def to_graph(lst):
    G = nx.Graph()
    for part in lst:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also implies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def to_edges(lst):
    """
        treat `l` as a Graph and return it's edges
        to_edges(['a','b','c','d']) -> [(a,b),(b,c),(c,d)]
    """
    if not len(lst):
        return
    it = iter(lst)
    last = next(it)
    for current in it:
        yield last, current
        last = current
