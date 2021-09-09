import pytest
import numpy as np

from ..utils import Cluster, gen_clusters


@pytest.mark.parametrize(
    'validation_matrix, val_clusters, val_unnasoc_inds',
    [
        # 3 tracks, no detections
        (
            np.array([[1],
                      [1],
                      [1]]),
            [],
            [0, 1, 2]
        ),
        # 3 tracks, 3 detections
        #  > 2 detected tracks, without shared detections, and 1 undetected track
        (
            np.array([[1, 1, 0, 1],
                      [1, 0, 1, 0],
                      [1, 0, 0, 0]]),
            [Cluster([0], [0, 1, 3]), Cluster([1], [0, 2])],
            [2]
        ),
        # 9 tracks, 7 detections
        #  > 7 detected tracks, with some shared detections, and 2 undetected tracks
        (
            np.array([[1, 1, 0, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 1, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1, 0, 0, 1],
                      [1, 0, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 1, 0],
                      [1, 0, 0, 0, 1, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0]]),
            [Cluster([0, 2, 3, 7], [0, 1, 3, 4, 7]), Cluster([1, 4], [0, 2, 5]), Cluster([6], [0, 6])],
            [5, 8]
        ),
        # 9 tracks, 7 detections
        #  > 7 detected tracks, with some shared detections, 2 undetected tracks and 2 non-gated detections
        (
            np.array([[1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                      [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                      [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
            [Cluster([0, 2, 3, 7], [0, 1, 3, 4, 8]), Cluster([1, 4], [0, 2, 6]), Cluster([6], [0, 7])],
            [5, 8]
        ),
    ]
)
def test_clustering(validation_matrix, val_clusters, val_unnasoc_inds):
    # Generate a random likelihood_matrix
    num_tracks, num_detections = validation_matrix.shape
    likelihood_matrix = np.zeros((num_tracks, num_detections))
    for i in range(num_tracks):
        liks = np.random.rand(num_detections)
        likelihood_matrix[i, :] = validation_matrix[i, :] * liks

    clusters, unnasoc_inds = gen_clusters(validation_matrix, likelihood_matrix)
    assert len(clusters) == len(val_clusters)
    for cluster, val_cluster in zip(clusters, val_clusters):
        assert cluster.tracks == val_cluster.tracks
        assert cluster.detections == val_cluster.detections
        assert np.array_equal(cluster.validation_matrix, validation_matrix[cluster.tracks, :][:, cluster.detections])
        assert np.array_equal(cluster.likelihood_matrix, likelihood_matrix[cluster.tracks, :][:, cluster.detections])
    assert set(unnasoc_inds) == set(val_unnasoc_inds)
