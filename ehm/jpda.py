# -*- coding: utf-8 -*-
from stonesoup.dataassociator.probability import JPDA
from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.numeric import Probability

from .utils import calc_validation_and_likelihood_matrices
from .ehm import EHM, EHM2


class JPDAWithEHM(JPDA):
    """ Joint Probabilistic Data Association with Efficient Hypothesis Management (EHM)

    This is a faster alternative of the standard :class:`~.JPDA` algorithm, which makes use of
    Efficient Hypothesis Management (EHM) to efficiently compute the joint associations. See
    Maskell et al. (2004) [1]_ for more details.

    .. [1] Maskell, S., Briers, M. and Wright, R., 2004, August. Fast mutual exclusion. In Signal and Data Processing
    of Small Targets 2004 (Vol. 5428, pp. 526-536). International Society for Optics and Photonics.
    """

    def associate(self, tracks, detections, timestamp, **kwargs):
        """"Associate tracks and detections

        Parameters
        ----------
        tracks : set of :class:`~.Track`
            Tracks which detections will be associated to.
        detections : set of :class:`~.Detection`
            Detections to be associated to tracks.
        timestamp : datetime.datetime
            Timestamp to be used for missed detections and to predict to.

        Returns
        -------
        : mapping of :class:`~.Track` : :class:`~.Hypothesis`
            Mapping of track to Hypothesis
        """

        # Calculate MultipleHypothesis for each Track over all
        # available Detections
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, timestamp)
            for track in tracks}

        multi_hypotheses = self._compute_multi_hypotheses(tracks, detections, hypotheses, timestamp)

        return multi_hypotheses

    @staticmethod
    def _run_ehm(validation_matrix, likelihood_matrix):
        return EHM.run(validation_matrix, likelihood_matrix)

    @classmethod
    def _compute_multi_hypotheses(cls, tracks, detections, hypotheses, time):

        # Tracks and detections must be in a list so we can keep track of their order
        track_list = list(tracks)
        detection_list = list(detections)

        # Calculate validation and likelihood matrices
        validation_matrix, likelihood_matrix = \
            calc_validation_and_likelihood_matrices(track_list, detection_list, hypotheses)

        # Run EHM
        assoc_prob_matrix = cls._run_ehm(validation_matrix, likelihood_matrix)

        # Calculate MultiMeasurementHypothesis for each Track over all
        # available Detections with probabilities drawn from the association matrix
        new_hypotheses = dict()

        for i, track in enumerate(track_list):

            single_measurement_hypotheses = list()

            # Null measurement hypothesis
            prob_misdetect = Probability(assoc_prob_matrix[i, 0])
            single_measurement_hypotheses.append(
                SingleProbabilityHypothesis(
                    hypotheses[track][0].prediction,
                    MissedDetection(timestamp=time),
                    measurement_prediction=hypotheses[track][0].measurement_prediction,
                    probability=prob_misdetect))

            # True hypotheses
            for hypothesis in hypotheses[track]:
                if not hypothesis:
                    continue

                # Get the detection index
                j = next(d_i+1 for d_i, detection in enumerate(detection_list)
                         if hypothesis.measurement == detection)

                pro_detect_assoc = Probability(assoc_prob_matrix[i, j])
                single_measurement_hypotheses.append(
                    SingleProbabilityHypothesis(
                        hypothesis.prediction,
                        hypothesis.measurement,
                        measurement_prediction=hypothesis.measurement_prediction,
                        probability=pro_detect_assoc))

            new_hypotheses[track] = MultipleHypothesis(single_measurement_hypotheses, True, 1)

        return new_hypotheses


class JPDAWithEHM2(JPDAWithEHM):
    """ Joint Probabilistic Data Association with Efficient Hypothesis Management 2 (EHM2)

    This is an enhanced version of the :class:`~.JPDAWithEHM` algorithm, that makes use of the Efficient Hypothesis
    Management 2 (EHM2) algorithm to efficiently compute the joint associations. See Horridge et al. (2006) [2]_ for
    more details.

    .. [2] Horridge, P. and Maskell, S., 2006, July. Real-time tracking of hundreds of targets with efficient exact
    JPDAF implementation. In 2006 9th International Conference on Information Fusion (pp. 1-8). IEEE.
    """

    @staticmethod
    def _run_ehm(validation_matrix, likelihood_matrix):
        return EHM2.run(validation_matrix, likelihood_matrix)
