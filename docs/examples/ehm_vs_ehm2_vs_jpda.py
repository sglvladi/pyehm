#!/usr/bin/env python
# coding: utf-8
"""
Comparison against standard JPDA
================================
"""

# %%
# Both :class:`~.EHM` and :class:`~.EHM2` provide an exact solution to the problem posed by the Joint Probabilistic
# Data Association (JPDA) algorithm. However, even though in the naive implementation JPDA the number of hypotheses,
# and, as a direct consequence, the time required to evaluate these, increases exponentially with number of targets
# and measurements, :class:`~.EHM` and :class:`~.EHM2` produce results in sum exponential time.
#
# In this example we shall compare the computational performance of the :class:`~.EHM` and :class:`~.EHM2`, against
# a naive implementation of JPDA, for a relatively dense scenario of 10 targets and 9 measurements. The
# validation and likelihood matrices for this scenario are defined below (For more information on how these matrices
# are defined, see the :ref:`sphx_glr_auto_examples_basic.py`):

import itertools
import datetime
import numpy as np

from pyehm.core import EHM, EHM2

validation_matrix = np.array([[1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
                              [1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
                              [1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                              [1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
                              [1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                              [1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                              [1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
                              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                              [1, 0, 0, 0, 1, 0, 1, 1, 1, 0]])

likelihood_matrix = np.array([[0.9, 0.13, 0.1, 0, 0.97, 0, 0.94, 0.55, 0, 0],
                              [0.55, 0.31, 0, 0.61, 0.27, 0.38, 0.34, 0.58, 0, 0],
                              [0.61, 0.55, 0, 0.32, 0, 0.25, 0.8, 0.94, 0.62, 0],
                              [0.45, 0.53, 0.61, 0.19, 0, 0, 0.95, 0.61, 0, 0.17],
                              [0.67, 0, 0.79, 0.99, 0, 0, 0, 0, 0.71, 0],
                              [0.51, 0.37, 0.04, 0, 0, 0.53, 0.92, 0.44, 0.95, 0],
                              [0.31, 0.03, 0, 0, 0, 0.08, 0.68, 0, 0.04, 0.31],
                              [0.23, 0.09, 0.21, 0, 0, 0, 0, 0, 0, 0],
                              [0.62, 0, 0, 0.19, 0.17, 0.31, 0.69, 0.89, 0.63, 0],
                              [0.44, 0, 0, 0, 0.53, 0, 0.49, 0.01, 0.31, 0]])

# %%
# Below we define the function ``jpda`` that computes the joint association probabilities based on the standard
# JPDA recursion that involves a full enumeration of all the joint hypotheses.

def jpda(validation_matrix, likelihood_matrix):
    num_tracks, num_detections = validation_matrix.shape

    possible_assoc = list()
    for track in range(num_tracks):
        track_possible_assoc = list()
        v_detections = np.flatnonzero(validation_matrix[track, :])
        for detection in v_detections:
            track_possible_assoc.append((track, detection))
        possible_assoc.append(track_possible_assoc)

    # Compute all possible joint hypotheses
    joint_hyps = itertools.product(*possible_assoc)

    # Compute valid joint hypotheses
    valid_joint_hypotheses = (joint_hypothesis for joint_hypothesis in joint_hyps if isvalidhyp(joint_hypothesis))

    # Compute likelihood for valid joint hypotheses
    valid_joint_hypotheses_lik = dict()
    for joint_hyp in valid_joint_hypotheses:
        lik = 1
        # The likelihood of a joint hypothesis is the product of the likelihoods of its member hypotheses
        for hyp in joint_hyp:
            track = hyp[0]
            detection = hyp[1]
            lik *= likelihood_matrix[track, detection]
        valid_joint_hypotheses_lik[joint_hyp] = lik

    # Compute the joint association probabilities
    assoc_matrix = np.zeros((num_tracks, num_detections))
    for track in range(num_tracks):
        v_detections = np.flatnonzero(validation_matrix[track, :])
        for detection in v_detections:
            # The joint assoc. probability for a track-detection hypothesis is the sum of the likelihoods of all
            # joint hypotheses that include this hypothesis
            prob = np.sum([lik for hyp, lik in valid_joint_hypotheses_lik.items() if (track, detection) in hyp])
            assoc_matrix[track, detection] = prob
        # Normalise
        assoc_matrix[track, :] /= np.sum(assoc_matrix[track, :])

    return assoc_matrix


def isvalidhyp(joint_hyp):
    used_detections = set()
    for hyp in joint_hyp:
        detection = hyp[1]
        if not detection:
            pass
        elif detection in used_detections:
            return False
        else:
            used_detections.add(detection)
    return True

# %%
# Now we can compare the above against :class:`~.EHM` and :class:`~.EHM2`, both in terms of accuracy and computation
# time. The accuracy comparison is just a safe-guard check to make sure that :class:`~.EHM` and :class:`~.EHM2`
# produce the same result as the standard JPDA.

# EHM
now = datetime.datetime.now()
assoc_matrix_ehm = EHM.run(validation_matrix, likelihood_matrix)
dt_ehm = datetime.datetime.now() - now

# EHM2
now = datetime.datetime.now()
assoc_matrix_ehm2 = EHM2.run(validation_matrix, likelihood_matrix)
dt_ehm2 = datetime.datetime.now() - now

# Standard JPDA
now = datetime.datetime.now()
assoc_matrix_jpda = jpda(validation_matrix, likelihood_matrix)
dt_jpda = datetime.datetime.now() - now

# Check if all results are the same
print(np.allclose(assoc_matrix_jpda, assoc_matrix_ehm, atol=1e-15)
      and np.allclose(assoc_matrix_jpda, assoc_matrix_ehm2, atol=1e-15))

# Compare the execution times
print('JPDA: {} seconds'.format(dt_jpda.total_seconds()))
print('EHM: {} seconds'.format(dt_ehm.total_seconds()))
print('EHM2: {} seconds'.format(dt_ehm2.total_seconds()))
