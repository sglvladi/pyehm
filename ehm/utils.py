import numpy as np
import networkx as nx
from networkx.algorithms.components.connected import connected_components


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