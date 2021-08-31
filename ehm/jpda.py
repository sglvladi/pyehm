# -*- coding: utf-8 -*-
import itertools
import numpy as np


from stonesoup.dataassociator.probability import JPDA
from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.numeric import Probability

from .ehm import EHM


class JPDAWithEHM(JPDA):
    """ Joint Probabilistic Data Association with Efficient Hypothesis Management
    This is a faster alternative of the standard :class:`~.JPDA` algorithm, which makes use of
    Efficient Hypothesis Management (EHM) to efficiently compute the joint associations. See
    Maskell et al. (2004) [#]_ for more details.
    .. [#] Simon Maskell, Mark Briers, Robert Wright, "Fast mutual exclusion," Proc. SPIE 5428,
           Signal and Data Processing of Small Targets 2004;
    """

    def associate(self, tracks, detections, timestamp, **kwargs):
        """Associate detections with predicted states.
        Parameters
        ----------
        tracks : list of :class:`Track`
            Current tracked objects
        detections : list of :class:`Detection`
            Retrieved measurements
        time : datetime
            Detection time to predict to
        Returns
        -------
        dict
            Key value pair of tracks with associated detection
        """

        # Calculate MultipleHypothesis for each Track over all
        # available Detections
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, timestamp)
            for track in tracks}

        multi_hypotheses = self._compute_multi_hypotheses(tracks, detections, hypotheses, timestamp)

        return multi_hypotheses

    def _compute_multi_hypotheses(self, tracks, detections, hypotheses, time):

        # Tracks and detections must be in a list so we can keep track of their order
        track_list = list(tracks)
        detection_list = list(detections)

        # Run EHM
        assoc_prob_matrix = EHM.run(track_list, detection_list, hypotheses)

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
