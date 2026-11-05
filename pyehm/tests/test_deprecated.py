"""Tests of the pyehm 2.2 API, deprecated in pyehm 2.3 and to be removed in pyehm 3.0."""
import warnings

import numpy as np
import pytest

from .jdpa_naive import jpda_naive, to_log
from ..core import EHM, EHM2
from ..utils import gen_clusters


def to_v22(likelihood_matrix):
    """The pyehm 2.2 validation and likelihood matrices for a likelihood matrix"""
    return (likelihood_matrix > 0).astype(int), likelihood_matrix


@pytest.mark.parametrize('ehm_class', [EHM, EHM2])
@pytest.mark.parametrize('scenario_name', ['scenarioA', 'scenarioB', 'scenarioC'])
def test_run(ehm_class, scenario_name, scenarios):
    validation_matrix, likelihood_matrix = to_v22(scenarios[scenario_name]['likelihood_matrix'])
    expected_result = jpda_naive(likelihood_matrix)
    calls = [lambda: ehm_class().run(validation_matrix, likelihood_matrix),
             lambda: ehm_class().run(validation_matrix, likelihood_matrix=likelihood_matrix),
             lambda: ehm_class().run(validation_matrix=validation_matrix, likelihood_matrix=likelihood_matrix)]
    for call in calls:
        with pytest.deprecated_call():
            result = call()
        assert np.allclose(result, expected_result)


@pytest.mark.parametrize('ehm_class', [EHM, EHM2])
def test_run_validation_matrix_takes_precedence(ehm_class, scenarios):
    # As in pyehm 2.2, likelihoods of associations that are not valid are ignored
    validation_matrix, likelihood_matrix = to_v22(scenarios['scenarioA']['likelihood_matrix'])
    noisy_likelihood_matrix = likelihood_matrix + 0.5 * (validation_matrix == 0)
    with pytest.deprecated_call():
        result = ehm_class().run(validation_matrix, noisy_likelihood_matrix)
    assert np.allclose(result, jpda_naive(likelihood_matrix))


@pytest.mark.parametrize('ehm_class', [EHM, EHM2])
@pytest.mark.parametrize('method', ['construct_net', 'construct_tree'])
def test_construct_validation_matrix(ehm_class, method, scenarios):
    likelihood_matrix = scenarios['scenarioA']['likelihood_matrix']
    validation_matrix, _ = to_v22(likelihood_matrix)
    expected = getattr(ehm_class(), method)(to_log(likelihood_matrix))
    for call in (lambda: getattr(ehm_class(), method)(validation_matrix),
                 lambda: getattr(ehm_class(), method)(validation_matrix.astype(float)),
                 lambda: getattr(ehm_class(), method)(validation_matrix=validation_matrix)):
        with pytest.deprecated_call():
            result = call()
        if method == 'construct_net':
            assert result.num_nodes == expected.num_nodes
        else:
            assert [n.track for n in result.nodes] == [n.track for n in expected.nodes]


@pytest.mark.parametrize('ehm_class', [EHM, EHM2])
def test_construct_net_log_likelihoods_do_not_warn(ehm_class, scenarios):
    log_likelihood_matrix = to_log(scenarios['scenarioA']['likelihood_matrix'])
    ones = np.ones(log_likelihood_matrix.shape)  # Only 1s, so it is detected as a validation matrix if positional
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ehm_class().construct_net(log_likelihood_matrix)
        ehm_class().construct_net(log_likelihood_matrix=ones)
    with pytest.deprecated_call():
        ehm_class().construct_net(ones)


@pytest.mark.parametrize('ehm_class', [EHM, EHM2])
def test_compute_association_probabilities(ehm_class, scenarios):
    likelihood_matrix = scenarios['scenarioA']['likelihood_matrix']
    log_likelihood_matrix = to_log(likelihood_matrix)
    algorithm = ehm_class()
    net = algorithm.construct_net(log_likelihood_matrix)
    expected_log_result = algorithm.compute_association_probabilities(net, log_likelihood_matrix=log_likelihood_matrix)
    assert np.allclose(np.exp(expected_log_result), jpda_naive(likelihood_matrix))
    for call in (lambda: algorithm.compute_association_probabilities(net, likelihood_matrix),
                 lambda: algorithm.compute_association_probabilities(net, likelihood_matrix=likelihood_matrix)):
        with pytest.warns(FutureWarning):
            result = call()
        assert np.allclose(result, np.exp(expected_log_result))


def test_gen_clusters(scenarios):
    likelihood_matrix = scenarios['scenarioB']['likelihood_matrix']
    validation_matrix, _ = to_v22(likelihood_matrix)
    expected = gen_clusters(to_log(likelihood_matrix))
    for call in (lambda: gen_clusters(validation_matrix, likelihood_matrix),
                 lambda: gen_clusters(validation_matrix=validation_matrix, likelihood_matrix=likelihood_matrix),
                 lambda: gen_clusters(validation_matrix)):
        with pytest.deprecated_call():
            clusters = call()
        assert [(c.tracks, c.detections) for c in clusters] == [(c.tracks, c.detections) for c in expected]

    with pytest.deprecated_call():
        clusters = gen_clusters(validation_matrix, likelihood_matrix)
    for cluster in clusters:
        if len(cluster.detections):
            with pytest.deprecated_call():
                assert np.array_equal(cluster.validation_matrix,
                                      validation_matrix[cluster.tracks, :][:, cluster.detections])
            with pytest.deprecated_call():
                # Exactly as given, as in pyehm 2.2
                assert np.array_equal(cluster.likelihood_matrix,
                                      likelihood_matrix[cluster.tracks, :][:, cluster.detections])

    # As in pyehm 2.2, clusters have empty likelihood matrices if only given a validation matrix
    with pytest.deprecated_call():
        clusters = gen_clusters(validation_matrix)
    for cluster in clusters:
        with pytest.deprecated_call():
            assert cluster.likelihood_matrix.size == 0


def test_net_validation_matrix(scenarios):
    likelihood_matrix = scenarios['scenarioA']['likelihood_matrix']
    validation_matrix, _ = to_v22(likelihood_matrix)
    net = EHM2().construct_net(to_log(likelihood_matrix))
    with pytest.deprecated_call():
        assert np.array_equal(net.validation_matrix, validation_matrix)


def test_new_api_types():
    # The deprecated API is added by subclassing, which must keep the classes usable as before
    assert issubclass(EHM, __import__('_pyehm').core.EHM)
    assert EHM.__name__ == 'EHM' and EHM2.__name__ == 'EHM2'
    assert 'run(log_likelihood_matrix' in EHM.run.__doc__
    assert 'run(validation_matrix' in EHM.run.__doc__
