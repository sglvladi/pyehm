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
    def _run_ehm(track_hypotheses):
        return EHM().run(track_hypotheses)

    @staticmethod
    def _calc_track_hypotheses(tracks, detections, hypotheses):
        """ Compute the detection ids and log-likelihoods of each track's hypotheses

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
        list of tuple of (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            A pair of arrays per track, containing the detection id and the log of the unnormalised
            likelihood of each of the track's hypotheses, in order. Detection id 0 corresponds to the
            null hypothesis.
        """

        # Detection ids start at 1, since 0 corresponds to the null hypothesis
        detection_ids = dict()
        for j, detection in enumerate(detections, start=1):
            detection_ids.setdefault(detection, j)

        # NOTE: Weights are converted via their log value, since converting them to float can
        # underflow to zero for very small likelihoods.
        track_hypotheses = list()
        for track in tracks:
            track_detections = [detection_ids[hyp.measurement] if hyp else 0 for hyp in hypotheses[track]]
            log_likelihoods = [Probability(hyp.weight).log_value for hyp in hypotheses[track]]
            track_hypotheses.append((np.array(track_detections, dtype=np.int32),
                                     np.array(log_likelihoods, dtype=float)))

        return track_hypotheses

    @classmethod
    def _compute_multi_hypotheses(cls, tracks, detections, hypotheses, time):

        # Tracks and detections must be in a list so we can keep track of their order
        track_list = list(tracks)
        detection_list = list(detections)

        # Calculate detection ids and log-likelihoods of each track's hypotheses
        track_hypotheses = cls._calc_track_hypotheses(track_list, detection_list, hypotheses)

        # Run EHM
        log_assoc_probs = cls._run_ehm(track_hypotheses)

        # Calculate MultiMeasurementHypothesis for each Track over all
        # available Detections with probabilities drawn from the association probabilities
        new_hypotheses = dict()

        for track, track_log_probs in zip(track_list, log_assoc_probs):

            single_measurement_hypotheses = list()
            hypotheses_and_probs = list(zip(hypotheses[track], track_log_probs))

            # Null measurement hypothesis
            null_hypothesis, log_prob_misdetect = next(
                (hyp, log_prob) for hyp, log_prob in hypotheses_and_probs if not hyp)
            single_measurement_hypotheses.append(
                SingleProbabilityHypothesis(
                    null_hypothesis.prediction,
                    MissedDetection(timestamp=time),
                    measurement_prediction=null_hypothesis.measurement_prediction,
                    probability=Probability(log_prob_misdetect, log_value=True)))

            # True hypotheses
            for hypothesis, log_prob in hypotheses_and_probs:
                if not hypothesis:
                    continue

                single_measurement_hypotheses.append(
                    SingleProbabilityHypothesis(
                        hypothesis.prediction,
                        hypothesis.measurement,
                        measurement_prediction=hypothesis.measurement_prediction,
                        probability=Probability(log_prob, log_value=True)))

            new_hypotheses[track] = MultipleHypothesis(single_measurement_hypotheses, True, 1)

        return new_hypotheses


class JPDAWithEHM2(JPDAWithEHM):
    """ Joint Probabilistic Data Association with Efficient Hypothesis Management 2 (EHM2)

    This is an enhanced version of the :class:`~.JPDAWithEHM` algorithm, that makes use of the Efficient Hypothesis
    Management 2 (EHM2) algorithm to efficiently compute the joint associations. See Horridge et al. (2006) [EHM2]_ for
    more details.
    """

    @staticmethod
    def _run_ehm(track_hypotheses):
        return EHM2().run(track_hypotheses)
