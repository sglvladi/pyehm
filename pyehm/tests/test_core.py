import pytest
import numpy as np

from .jdpa_naive import gen_random_scenario, jpda_naive
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
    validation_matrix = scenarios[scenario_name]['validation_matrix']
    likelihood_matrix = scenarios[scenario_name]['likelihood_matrix']

    # Read expected results
    expected_result = jpda_naive(validation_matrix, likelihood_matrix)

    # Run EHM
    result = ehm_class.run(validation_matrix, likelihood_matrix)
    assert np.allclose(result, expected_result)


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
        validation_matrix, likelihood_matrix = gen_random_scenario(num_tracks, num_detections)
        expected_result = jpda_naive(validation_matrix, likelihood_matrix)
        ehm1_result = EHM.run(validation_matrix, likelihood_matrix)
        ehm2_result = EHM2.run(validation_matrix, likelihood_matrix)
        assert np.allclose(expected_result, ehm1_result)
        assert np.allclose(expected_result, ehm2_result)