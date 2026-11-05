"""Support for the pyehm 2.2 API, which took validation matrices and (linear) likelihood matrices.

Deprecated in pyehm 2.3 and to be removed in pyehm 3.0, along with this module.
"""
import warnings
import weakref

import numpy as np
from _pyehm.net import EHMNet
from _pyehm.utils import Cluster

REMOVAL = 'pyehm 3.0'


def warn_deprecated(message, category=DeprecationWarning):
    # stacklevel=3 attributes the warning to the code calling the deprecated function
    warnings.warn(f'{message} This will be removed in {REMOVAL}.', category, stacklevel=3)


def bind(args, kwargs, names, defaults=None):
    """Get the values of the given argument names from positional and keyword arguments."""
    if len(args) > len(names):
        raise TypeError(f'Expected at most {len(names)} arguments, got {len(args)}')
    values = dict(defaults or {})
    values.update(zip(names, args))
    for name, value in kwargs.items():
        if name not in names:
            raise TypeError(f"Unexpected keyword argument '{name}'")
        values[name] = value
    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(f'Missing argument(s): {", ".join(missing)}')
    return [values[name] for name in names]


def is_validation_matrix(matrix):
    """Whether a matrix contains only 0s and 1s, i.e. is a pyehm 2.2 validation matrix.

    A log-likelihood matrix with only 0s and 1s would require every likelihood to be exactly 1 or e, with no invalid
    (``-inf``) associations, so such a matrix is assumed to be a validation matrix.
    """
    try:
        matrix = np.asarray(matrix)
    except ValueError:  # E.g. ragged per-track hypotheses
        return False
    if matrix.ndim != 2 or matrix.size == 0 or not (np.issubdtype(matrix.dtype, np.number) or matrix.dtype == bool):
        return False
    return bool(np.isin(matrix, (0, 1)).all())


def to_log_likelihoods(validation_matrix, likelihood_matrix=None):
    """Convert a pyehm 2.2 validation matrix and likelihood matrix to a log-likelihood matrix."""
    valid = np.asarray(validation_matrix) != 0
    if likelihood_matrix is None or np.size(likelihood_matrix) == 0:
        log_likelihoods = np.zeros(valid.shape)
    else:
        with np.errstate(divide='ignore'):
            log_likelihoods = np.log(np.asarray(likelihood_matrix, dtype=float))
    return np.where(valid, log_likelihoods, -np.inf)


def add_deprecation_note(doc, note):
    """Insert a deprecation note after the summary of a docstring that starts with a signature line."""
    lines = (doc or '').split('\n')
    end_of_summary = next((i for i, line in enumerate(lines[1:], start=1) if not line.strip()), len(lines))
    note_lines = ['', '.. deprecated:: 2.3', *(f'    {line}' for line in note.split('\n'))]
    return '\n'.join(lines[:end_of_summary] + note_lines + lines[end_of_summary:])


# pyehm 2.2 attributes of nets and clusters
def _net_validation_matrix(self):
    warn_deprecated('EHMNet.validation_matrix is deprecated; use EHMNet.log_likelihood_matrix, where -inf denotes '
                    'invalid associations.')
    return (self.log_likelihood_matrix != -np.inf).astype(int)


def _cluster_validation_matrix(self):
    warn_deprecated('Cluster.validation_matrix is deprecated; use Cluster.log_likelihood_matrix, where -inf denotes '
                    'invalid associations.')
    return (self.log_likelihood_matrix != -np.inf).astype(int)


# The exact likelihood matrices of clusters generated from pyehm 2.2 arguments, since exp(log(x)) may not equal x
_likelihood_matrices = weakref.WeakKeyDictionary()


def remember_likelihood_matrices(clusters, likelihood_matrix):
    """Remember the likelihood matrices of clusters generated from a pyehm 2.2 likelihood matrix (or none)."""
    for cluster in clusters:
        if likelihood_matrix is None or np.size(likelihood_matrix) == 0 or not len(cluster.detections):
            _likelihood_matrices[cluster] = np.zeros((0, 0))
        else:
            _likelihood_matrices[cluster] = np.asarray(likelihood_matrix, dtype=float)[
                np.ix_(cluster.tracks, cluster.detections)]
    return clusters


def _cluster_likelihood_matrix(self):
    warn_deprecated('Cluster.likelihood_matrix is deprecated; use Cluster.log_likelihood_matrix.')
    if self in _likelihood_matrices:
        return _likelihood_matrices[self]
    return np.exp(self.log_likelihood_matrix)


EHMNet.validation_matrix = property(_net_validation_matrix, doc=(
    'Indicator matrix of the valid associations of the net.\n\n'
    '.. deprecated:: 2.3\n    Will be removed in pyehm 3.0. Use :attr:`log_likelihood_matrix` instead, where '
    '``-numpy.inf`` denotes invalid associations.'))
Cluster.validation_matrix = property(_cluster_validation_matrix, doc=(
    'Indicator matrix of the valid associations of the cluster.\n\n'
    '.. deprecated:: 2.3\n    Will be removed in pyehm 3.0. Use :attr:`log_likelihood_matrix` instead, where '
    '``-numpy.inf`` denotes invalid associations.'))
Cluster.likelihood_matrix = property(_cluster_likelihood_matrix, doc=(
    'Likelihood matrix of the cluster.\n\n'
    '.. deprecated:: 2.3\n    Will be removed in pyehm 3.0. Use :attr:`log_likelihood_matrix` instead.'))
