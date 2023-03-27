# -*- coding: utf-8 -*-
import numpy as np
from stonesoup.dataassociator.probability import JPDA
from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.numeric import Probability

from pyehm.core import EHM, EHM2


class JPDAWithEHM(JPDA):
    """ Joint Probabilistic Data Association with Efficient Hypothesis Management (EHM)

    This is a faster alternative of the standard :class:`~.JPDA` algorithm, which makes use of
    Efficient Hypothesis Management (EHM) to efficiently compute the joint associations. See
    Maskell et al. (2004) [EHM1]_ for more details.
    """

    def associate(self, tracks, detections, timestamp, **kwargs):
        """Associate tracks and detections

        Parameters
        ----------
        tracks : set of :class:`stonesoup.types.track.Track`
            Tracks which detections will be associated to.
        detections : set of :class:`stonesoup.types.detection.Detection`
            Detections to be associated to tracks.
        timestamp : :class:`datetime.datetime`
            Timestamp to be used for missed detections and to predict to.

        Returns
        -------
        : mapping of :class:`stonesoup.types.track.Track` : :class:`stonesoup.types.hypothesis.Hypothesis`
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

    @staticmethod
    def _calc_validation_and_likelihood_matrices(tracks, detections, hypotheses):
        """ Compute the validation and likelihood matrices

        Parameters
        ----------
        tracks: list of :class:`stonesoup.types.track.Track`
            Current tracked objects
        detections : list of :class:`stonesoup.types.detection.Detection`
            Retrieved measurements
        hypotheses: dict
            Key value pairs of tracks with associated detections

        Returns
        -------
        :class:`numpy.ndarray`
            An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
            (aka. valid) associations between tracks and detections. The first column corresponds
            to the null hypothesis (hence contains all ones).
        :class:`numpy.ndarray`
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

        return validation_matrix.astype(int), likelihood_matrix.astype(float)

    @classmethod
    def _compute_multi_hypotheses(cls, tracks, detections, hypotheses, time):

        # Tracks and detections must be in a list so we can keep track of their order
        track_list = list(tracks)
        detection_list = list(detections)

        # Calculate validation and likelihood matrices
        validation_matrix, likelihood_matrix = \
            cls._calc_validation_and_likelihood_matrices(track_list, detection_list, hypotheses)

        # Run EHM
        assoc_prob_matrix = cls._run_ehm(validation_matrix, likelihood_matrix)

        # Calculate MultiMeasurementHypothesis for each Track over all
        # available Detections with probabilities drawn from the association matrix
        new_hypotheses = dict()

        for i, track in enumerate(track_list):

            single_measurement_hypotheses = list()

            # Null measurement hypothesis
            null_hypothesis = next((hyp for hyp in hypotheses[track] if not hyp), None)
            prob_misdetect = Probability(assoc_prob_matrix[i, 0])
            single_measurement_hypotheses.append(
                SingleProbabilityHypothesis(
                    null_hypothesis.prediction,
                    MissedDetection(timestamp=time),
                    measurement_prediction=null_hypothesis.measurement_prediction,
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
    Management 2 (EHM2) algorithm to efficiently compute the joint associations. See Horridge et al. (2006) [EHM2]_ for
    more details.
    """

    @staticmethod
    def _run_ehm(validation_matrix, likelihood_matrix):
        assoc_prob_matrix = EHM2.run(validation_matrix, likelihood_matrix)
        return assoc_prob_matrix
