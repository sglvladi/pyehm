import pytest
import numpy as np

from .jdpa_naive import gen_random_scenario, jpda_naive, to_log, to_track_hypotheses
from ..core import EHM, EHM2
from ..utils import gen_clusters


@pytest.mark.parametrize(
    'scenario_name, ehm_class',
    [
        ('scenarioA', EHM),
        ('scenarioB', EHM),
        ('scenarioC', EHM),
        ('scenarioA', EHM2),
        ('scenarioB', EHM2),
        ('scenarioC', EHM2)
    ]
)
def test_ehm_run(scenario_name, ehm_class, scenarios):
    likelihood_matrix = scenarios[scenario_name]['likelihood_matrix']

    # Read expected results
    expected_result = jpda_naive(likelihood_matrix)

    # Run EHM
    log_result = ehm_class().run(to_log(likelihood_matrix))
    assert np.allclose(np.exp(log_result), expected_result)


@pytest.mark.parametrize(
    'num_tracks, num_detections',
    [
        (5, 0),
        (0, 5),
        (0, 0),
        (4, 4),
        (8, 9),
    ]
)
def test_ehm_run_random(num_tracks, num_detections):
    for _ in range(10):
        likelihood_matrix = gen_random_scenario(num_tracks, num_detections)
        expected_result = jpda_naive(likelihood_matrix)
        ehm1_result = np.exp(EHM().run(to_log(likelihood_matrix)))
        ehm2_result = np.exp(EHM2().run(to_log(likelihood_matrix)))
        assert np.allclose(expected_result, ehm1_result)
        assert np.allclose(expected_result, ehm2_result)


@pytest.mark.parametrize('ehm_class', [EHM, EHM2])
def test_ehm_run_underflow(ehm_class):
    # Likelihoods this small underflow to zero when multiplied in linear space
    np.random.seed(0)
    likelihood_matrix = gen_random_scenario(10, 10)
    log_likelihood_matrix = to_log(likelihood_matrix)
    small_log_likelihood_matrix = log_likelihood_matrix - 700

    # Shifting each row by a constant scales every joint hypothesis equally, so the result must not change
    expected_result = ehm_class().run(log_likelihood_matrix)
    result = ehm_class().run(small_log_likelihood_matrix)
    assert np.all(np.isfinite(np.exp(result).sum(axis=1)))
    assert np.allclose(np.exp(result), np.exp(expected_result))


@pytest.mark.parametrize('ehm_class', [EHM, EHM2])
@pytest.mark.parametrize(
    'num_tracks, num_detections',
    [
        (5, 0),
        (0, 5),
        (0, 0),
        (4, 4),
        (8, 9),
    ]
)
def test_ehm_run_hypotheses(ehm_class, num_tracks, num_detections):
    for _ in range(10):
        likelihood_matrix = gen_random_scenario(num_tracks, num_detections)
        expected_result = ehm_class().run(to_log(likelihood_matrix))
        track_hypotheses = to_track_hypotheses(likelihood_matrix)
        result = ehm_class().run(track_hypotheses)
        assert len(result) == num_tracks
        for track, (detections, _) in enumerate(track_hypotheses):
            assert np.array_equal(result[track], expected_result[track, detections])


@pytest.mark.parametrize('ehm_class', [EHM, EHM2])
def test_ehm_run_hypotheses_arbitrary_ids(ehm_class, scenarios):
    # Results must not depend on the values of the detection ids or the order of each track's hypotheses
    likelihood_matrix = scenarios['scenarioB']['likelihood_matrix']
    expected_result = np.exp(ehm_class().run(to_log(likelihood_matrix)))

    track_hypotheses = list()
    for detections, log_likelihoods in to_track_hypotheses(likelihood_matrix):
        order = np.argsort(-detections)
        track_hypotheses.append((detections[order] * 1000, log_likelihoods[order]))

    for track, log_probs in enumerate(ehm_class().run(track_hypotheses)):
        detections = track_hypotheses[track][0] // 1000
        assert np.allclose(np.exp(log_probs), expected_result[track, detections])


@pytest.mark.parametrize('ehm_class', [EHM, EHM2])
def test_ehm_run_hypotheses_invalid_entries(ehm_class):
    # Hypotheses with a log-likelihood of -inf must behave as if they were not listed
    track_hypotheses = [(np.array([0, 1]), np.log([0.2, 0.8])), (np.array([0, 2]), np.log([0.5, 0.5]))]
    with_invalid = [(np.array([0, 1, 2]), np.array([np.log(0.2), np.log(0.8), -np.inf])),
                    (np.array([0, 2, 1]), np.array([np.log(0.5), np.log(0.5), -np.inf]))]
    result = ehm_class().run(track_hypotheses)
    result_with_invalid = ehm_class().run(with_invalid)
    for track in range(2):
        assert np.array_equal(result_with_invalid[track][:2], result[track])
        assert result_with_invalid[track][2] == -np.inf


@pytest.mark.parametrize('ehm_class', [EHM, EHM2])
@pytest.mark.parametrize(
    'track_hypotheses',
    [
        [(np.array([0, 1]), np.array([0.]))],             # Mismatched lengths
        [(np.array([0, -1]), np.array([0., 0.]))],        # Negative id
        [(np.array([0, 1, 1]), np.array([0., 0., 0.]))],  # Duplicate id
    ]
)
def test_ehm_run_hypotheses_bad_input(ehm_class, track_hypotheses):
    with pytest.raises(ValueError):
        ehm_class().run(track_hypotheses)


@pytest.mark.parametrize('ehm_class', [EHM, EHM2])
def test_ehm_run_hypotheses_many_tracks(ehm_class):
    # Pairs of tracks that share a detection. A dense matrix of this size would need 80GB of memory.
    num_pairs = 50000
    log_pair = np.log([[0.2, 0.8], [0.4, 0.6]])
    track_hypotheses = list()
    for pair in range(num_pairs):
        detections = np.array([0, 2 * pair + 1], dtype=np.int32)
        track_hypotheses += [(detections, log_pair[0]), (detections, log_pair[1])]

    expected_result = ehm_class().run(log_pair)
    result = ehm_class().run(track_hypotheses)
    assert len(result) == 2 * num_pairs
    for track in (0, 1, 2 * num_pairs - 2, 2 * num_pairs - 1):
        assert np.allclose(result[track], expected_result[track % 2, :])


def test_construct_net_requires_single_cluster():
    log_likelihood_matrix = to_log(np.array([[0.2, 0.8, 0], [0.4, 0, 0.6]]))  # Tracks share no detections
    with pytest.raises(ValueError, match='gen_clusters'):
        EHM2().construct_net(log_likelihood_matrix)
    with pytest.raises(ValueError, match='gen_clusters'):
        EHM2().construct_tree(log_likelihood_matrix)


@pytest.mark.parametrize('ehm_class', [EHM, EHM2])
def test_construct_net_requires_tracks(ehm_class):
    with pytest.raises(ValueError):
        ehm_class().construct_net(np.zeros((0, 0)))


@pytest.mark.parametrize('ehm_class', [EHM, EHM2])
def test_construct_net_per_cluster(ehm_class):
    # Constructing a net per cluster, as instructed by the docs, must match the result of run
    likelihood_matrix = np.array([[0.2, 0.8, 0, 0, 0],
                                  [0.4, 0, 0.6, 0, 0],
                                  [0.3, 0.3, 0, 0.4, 0],
                                  [1, 0, 0, 0, 0]])
    log_likelihood_matrix = to_log(likelihood_matrix)

    result = np.full(likelihood_matrix.shape, -np.inf)
    for cluster in gen_clusters(log_likelihood_matrix):
        if len(cluster.detections):
            algorithm = ehm_class()
            net = algorithm.construct_net(cluster.log_likelihood_matrix)
            cluster_result = algorithm.compute_association_probabilities(
                net, log_likelihood_matrix=cluster.log_likelihood_matrix)
            result[np.ix_(cluster.tracks, cluster.detections)] = cluster_result
        else:
            result[cluster.tracks, 0] = 0
    assert np.array_equal(result, ehm_class().run(log_likelihood_matrix))
