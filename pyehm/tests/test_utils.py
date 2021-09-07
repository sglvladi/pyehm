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
            [Cluster({0}, {1, 3}), Cluster({1}, {2})],
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
            [Cluster({0, 2, 3, 7}, {1, 3, 4, 7}), Cluster({1, 4}, {2, 5}), Cluster({6}, {6})],
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
            [Cluster({0, 2, 3, 7}, {1, 3, 4, 8}), Cluster({1, 4}, {2, 6}), Cluster({6}, {7})],
            [5, 8]
        ),
    ]
)
def test_clustering(validation_matrix, val_clusters, val_unnasoc_inds):
    clusters, unnasoc_inds = gen_clusters(validation_matrix)
    assert len(clusters) == len(val_clusters)
    for cluster, val_cluster in zip(clusters, val_clusters):
        assert cluster.rows == val_cluster.rows
        assert cluster.cols == val_cluster.cols
    assert set(unnasoc_inds) == set(val_unnasoc_inds)
