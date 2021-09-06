# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.algorithms.components.connected import connected_components


class EHMNetNode:
    def __init__(self, layer, detections=None, v_detections=None, remainders=None):
        # Index of the layer (track) in the network
        self.layer = layer
        # Set of detection indices considered down to current node
        self.detections = detections if detections is not None else set()
        # Set of detections that can be considered by current node
        self.v_detections = v_detections if v_detections else set()
        # Set of remaining detection indices to be considered up to the next layer
        self.remainders = remainders if remainders is not None else set()
        # Index of the node when added to the network. This is set by the network and
        # should not be edited.
        self.ind = None

    def __repr__(self):
        return 'EHMNetNode(ind={}, layer={})'.format(self.ind, self.layer)


class EHM2NetNode(EHMNetNode):
    def __init__(self, layer, track=None, subnet=0, detections=None, remainders=None, v_detections=None):
        super().__init__(layer, detections, v_detections, remainders)
        # Index of track this node relates to
        self.track = track
        # Index of subnet the node belongs to
        self.subnet = subnet

    def __repr__(self):
        return 'EHM2NetNode(ind={}, layer={}, track={}, subnet={})'.format(self.ind, self.layer, self.track, self.subnet)


class EHMNet:
    def __init__(self, nodes, edges=None, validation_matrix=None):
        for n_i, node in enumerate(nodes):
            node.ind = n_i
        self._nodes = nodes
        self.validation_matrix = validation_matrix
        self.edges = edges if edges is not None else dict()
        self.parents_per_detection = dict()
        self.children_per_detection = dict()
        self.nodes_per_track = dict()

    @property
    def root(self):
        return self.nodes[0]

    @property
    def num_nodes(self):
        return len(self._nodes)

    @property
    def nodes(self):
        return self._nodes

    @property
    def nodes_forward(self):
        return sorted(self.nodes, key=lambda x: x.layer)

    def add_node(self, node, parent, identity):
        # Set the node index
        node.ind = len(self.nodes)
        # Add node to graph
        self.nodes.append(node)
        # Create edge from parent to child
        self.edges[(parent, node)] = {identity}
        # Create parent-child-detection look-up
        self.parents_per_detection[(node, identity)] = {parent}
        if (parent, identity) in self.children_per_detection:
            self.children_per_detection[(parent, identity)].add(node)
        else:
            self.children_per_detection[(parent, identity)] = {node}

    def add_edge(self, parent, child, identity):
        if (parent, child) in self.edges:
            self.edges[(parent, child)].add(identity)
        else:
            self.edges[(parent, child)] = {identity}
        if (child, identity) in self.parents_per_detection:
            self.parents_per_detection[(child, identity)].add(parent)
        else:
            self.parents_per_detection[(child, identity)] = {parent}
        if (parent, identity) in self.children_per_detection:
            self.children_per_detection[(parent, identity)].add(child)
        else:
            self.children_per_detection[(parent, identity)] = {child}

    def get_parents(self, node):
        return [edge[0] for edge in self.edges if edge[1] == node]

    def get_children(self, node):
        return [edge[1] for edge in self.edges if edge[0] == node]

    @property
    def nx_graph(self):
        g = nx.Graph()
        for child in sorted(self.nodes, key= lambda x: x.layer):
            parents = self.get_parents(child)
            if isinstance(child, EHM2NetNode):
                track = child.track
            else:
                track = child.layer
            if track > -1:
                v_dets = set(np.flatnonzero(self.validation_matrix[track, :]))
            else:
                v_dets = set()
            measset = v_dets - child.v_detections
            g.add_node(child.ind, track=track, measset=measset)
            for parent in parents:
                #label = ','.join([str(s) for s in self.edges[(parent, child)]])
                label = str(self.edges[(parent, child)]).replace('{','').replace('}','')
                g.add_edge(parent.ind, child.ind, detections=label)
        return g

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()
        g = self.nx_graph
        pos = graphviz_layout(g, prog="dot")
        nx.draw(g, pos, ax=ax, node_size=0)
        labels = dict()
        for n in g.nodes:
            t = g.nodes[n]['track']
            s = str(g.nodes[n]['measset']) if len(g.nodes[n]['measset']) else 'Ø'
            if t > -1:
                labels[n] = '{{{}, {}}}'.format(t, s)
            else:
                labels[n] = 'Ø'
        pos_labels = {}
        for node, coords in pos.items():
            pos_labels[node] = (coords[0] + 10, coords[1])
        nx.draw_networkx_labels(g, pos_labels, ax=ax, labels=labels, horizontalalignment='left')
        edge_labels = nx.get_edge_attributes(g, 'detections')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)


class Tree:
    def __init__(self, track, children, detections, subtree):
        self.track = track
        self.children = children
        self.detections = detections
        self.subtree = subtree

    @property
    def depth(self):
        depth = 1
        c_depth = 0
        for child in self.children:
            child_depth = child.depth
            if child_depth > c_depth:
                c_depth = child_depth
        return depth + c_depth

    @property
    def nodes(self):
        nodes = [self]
        for child in self.children:
            nodes += child.nodes
        return nodes

    @property
    def nx_graph(self):
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

    def plot(self, ax=None):
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
    def __init__(self, rows=None, cols=None):
        self.rows = set(rows) if rows is not None else set()
        self.cols = set(cols) if cols is not None else set()


def calc_validation_and_likelihood_matrices(tracks, detections, hypotheses):
    """ Compute the validation and likelihood matrices

    Parameters
    ----------
    tracks: list of :class:`Track`
        Current tracked objects
    detections : list of :class:`Detection`
        Retrieved measurements
    hypotheses: dict
        Key value pairs of tracks with associated detections

    Returns
    -------
    :class:`np.array`
        An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
        (aka. valid) associations between tracks and detections. The first column corresponds
        to the null hypothesis (hence contains all ones).
    :class:`np.array`
        A matrix of shape (num_tracks, num_detections + 1) containing the unnormalised
        likelihoods for all combinations of tracks and detections. The first column corresponds
        to the null hypothesis.
    """

    # Ensure tracks and detections are lists (not sets)
    tracks, detections = list(tracks), list(detections)

    # Construct validation and likelihood matrices
    # Both matrices have shape (num_tracks, num_detections + 1), where the first column
    # corresponds to the null hypothesis.
    num_tracks, num_detections = len(tracks), len(detections)
    likelihood_matrix = np.zeros((num_tracks, num_detections + 1))
    for i, track in enumerate(tracks):
        for hyp in hypotheses[track]:
            if not hyp:
                likelihood_matrix[i, 0] = hyp.weight
            else:
                j = next(d_i for d_i, detection in enumerate(detections)
                         if hyp.measurement == detection)
                likelihood_matrix[i, j + 1] = hyp.weight
    validation_matrix = likelihood_matrix > 0
    return validation_matrix, likelihood_matrix


def to_graph(l):
    G = nx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def to_edges(l):
    """
        treat `l` as a Graph and return it's edges
        to_edges(['a','b','c','d']) -> [(a,b),(b,c),(c,d)]
    """
    if not len(l):
        return
    it = iter(l)
    last = next(it)
    for current in it:
        yield last, current
        last = current


def gen_clusters(v_matrix):
    """ Cluster tracks into groups that sharing detections

    Parameters
    ----------
    v_matrix: :class:`np.array`
        An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
        (aka. valid) associations between tracks and detections. The first column corresponds
        to the null hypothesis (hence contains all ones).

    Returns
    -------
    list of :class:`Cluster` objects
        A list of :class:`Cluster` objects, where each cluster contains the indices of the rows
        (tracks) and columns (detections) pertaining to the cluster
    list of int
        A list of row (track) indices that have not been associated to any detections
    """

    # Validation matrix for all detections except null
    v_matrix_true = v_matrix[:, 1:]

    # Initiate parameters
    num_rows, num_cols = np.shape(v_matrix_true)  # Number of tracks

    # Form clusters of tracks sharing measurements
    unassoc_rows = set([i for i in range(num_rows)])
    clusters = list()

    # List of tracks gated for each detection
    v_lists = [np.flatnonzero(v_matrix_true[:, col_ind]) for col_ind in range(num_cols)]

    # Get clusters of tracks sharing common detections
    G = to_graph(v_lists)
    cluster_rows = [t for t in connected_components(G)]

    # Create cluster objects that contain the indices of tracks (rows) and detections (cols)
    for rows in cluster_rows:
        v_cols = set()
        for row_ind in rows:
            v_cols |= set(np.flatnonzero(v_matrix_true[row_ind, :])+1)
        clusters.append(Cluster(rows, v_cols))

    # Get tracks (rows) that are not associated to any detections
    assoc_rows = set([j for i in cluster_rows for j in i])
    unassoc_rows = unassoc_rows - assoc_rows

    return clusters, list(unassoc_rows)