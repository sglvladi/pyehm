import pytest
import numpy as np

from .jdpa_naive import gen_random_scenario, jpda_naive, to_log
from ..core import EHM, EHM2


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
