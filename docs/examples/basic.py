#!/usr/bin/env python
# coding: utf-8
"""
Basic Example
=============
"""

import numpy as np

# %%
# Formulating the possible associations between targets and measurements
# ----------------------------------------------------------------------
#
# Both :class:`~.EHM` and :class:`~.EHM2` operate on a ``validation_matrix`` and a ``likelihood_matrix``. The
# ``validation_matrix`` is an indicator matrix that represents the possible associations between different targets
# and measurements, while the ``likelihood_matrix`` contains the respective likelihoods/probabilities of these
# associations. Both matrices have a shape ``(N_T, N_M+1)``, where ``N_T`` is the number of targets and ``N_M`` is
# the numer of measurements.
#
# For example, assume we have the following scenario of 4 targets and 4 measurements (taken from Section 4.4 of
# [EHM2]_):
#
# +---------------+---------------------------+
# | Target index  | Gated measurement indices |
# +===============+===========================+
# | 0             | 0, 1                      |
# +---------------+---------------------------+
# | 1             | 0, 1, 2, 3                |
# +---------------+---------------------------+
# | 2             | 0, 1, 2                   |
# +---------------+---------------------------+
# | 3             | 0, 3, 4                   |
# +---------------+---------------------------+
#
# where the null measurement hypothesis is given the index of 0. Then the ``validation_matrix`` would be a ``(4, 5)``
# numpy array of the following form:

validation_matrix = np.array([[1, 1, 0, 0, 0],  # 0 -> 0,1
                              [1, 1, 1, 1, 0],  # 1 -> 0,1,2,3
                              [1, 1, 1, 0, 0],  # 2 -> 0,1,2
                              [1, 0, 0, 1, 1]]) # 3 -> 0,3,4

# %%
# The ``likelihood_matrix`` is such that each element ``likelihood_matrix[i, j]`` contains the respective likelihood
# of target ``i`` being associated to measurement ``j``. Therefore, based on the above example, the
# ``likelihood_matrix`` could be the following:

likelihood_matrix = np.array([[0.1, 0.9, 0, 0, 0],
                              [0.1, 0.3, 0.2, 0.4, 0],
                              [0.7, 0.1, 0.2, 0, 0],
                              [0.2, 0, 0, 0.75, 0.05]])

# %%
# Computing joint association probabilities
# -----------------------------------------
# Based on the above, we can use :class:`~.EHM` or :class:`~.EHM2` to compute the joint association probabilities
# matrix ``assoc_matrix`` as follows:

from pyehm.core import EHM, EHM2

assoc_matrix_ehm = EHM.run(validation_matrix, likelihood_matrix)
print('assoc_matrix_ehm =\n {}\n'.format(assoc_matrix_ehm))
# or
assoc_matrix_ehm2 = EHM2.run(validation_matrix, likelihood_matrix)
print('assoc_matrix_ehm2 =\n {}'.format(assoc_matrix_ehm2))

# %%
# Note that both :class:`~.EHM` and :class:`~.EHM2` should produce the same results, although :class:`~.EHM2` should,
# in principle, be significantly faster for large numbers of targets and measurements.

# Check if the probability matrices produced by EHM and EHM2 are equal
print(np.allclose(assoc_matrix_ehm, assoc_matrix_ehm2))