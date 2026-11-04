import datetime

import numpy as np
import pytest
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from .jdpa_naive import jpda_naive, to_log
from ..plugins.stonesoup import JPDAWithEHM, JPDAWithEHM2


def gen_hypotheses(tracks, detections, log_likelihood_matrix, timestamp):
    hypotheses = dict()
    for track, track_log_likelihoods in zip(tracks, log_likelihood_matrix):
        prediction = GaussianStatePrediction([[0]], [[1]], timestamp=timestamp)
        track_hypotheses = [SingleProbabilityHypothesis(prediction, MissedDetection(timestamp=timestamp),
                                                        Probability(track_log_likelihoods[0], log_value=True))]
        for detection, log_likelihood in zip(detections, track_log_likelihoods[1:]):
            if log_likelihood != -np.inf:
                track_hypotheses.append(SingleProbabilityHypothesis(prediction, detection,
                                                                    Probability(log_likelihood, log_value=True)))
        hypotheses[track] = MultipleHypothesis(track_hypotheses)
    return hypotheses


@pytest.mark.parametrize('associator', [JPDAWithEHM, JPDAWithEHM2])
@pytest.mark.parametrize('log_scale', [0, -2000])
def test_compute_multi_hypotheses(associator, log_scale, scenarios):
    likelihood_matrix = scenarios['scenarioA']['likelihood_matrix']
    num_tracks, num_columns = likelihood_matrix.shape
    timestamp = datetime.datetime(2026, 1, 1)
    tracks = [Track([GaussianState([[0]], [[1]], timestamp=timestamp)]) for _ in range(num_tracks)]
    detections = [Detection([[j]], timestamp=timestamp) for j in range(num_columns - 1)]

    # A log_scale of -2000 makes the weights underflow to zero if they are converted to float
    log_likelihood_matrix = to_log(likelihood_matrix) + log_scale
    hypotheses = gen_hypotheses(tracks, detections, log_likelihood_matrix, timestamp)
    multi_hypotheses = associator._compute_multi_hypotheses(tracks, detections, hypotheses, timestamp)

    expected_result = jpda_naive(likelihood_matrix)
    for i, track in enumerate(tracks):
        result = np.zeros(num_columns)
        for hypothesis in multi_hypotheses[track]:
            j = detections.index(hypothesis.measurement) + 1 if hypothesis else 0
            result[j] = float(hypothesis.probability)
        assert np.allclose(result, expected_result[i, :])
