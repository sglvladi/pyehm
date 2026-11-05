import numpy as np
from _pyehm.core import EHM as _EHM, EHM2 as _EHM2

from ._compat import add_deprecation_note, bind, is_validation_matrix, to_log_likelihoods, warn_deprecated

_RUN_DEPRECATED_DOC = '''
run(validation_matrix: numpy.ndarray, likelihood_matrix: numpy.ndarray) -> numpy.ndarray
Run {algorithm} with a validation matrix and a likelihood matrix, as in pyehm 2.2

.. deprecated:: 2.3
    Will be removed in pyehm 3.0. Pass a log-likelihood matrix instead, i.e. ``numpy.log(likelihood_matrix)``
    with ``-numpy.inf`` wherever ``validation_matrix`` is 0, and take ``numpy.exp`` of the returned
    log-probabilities.

Parameters
----------
validation_matrix: :class:`numpy.ndarray`
    An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
    (aka. valid) associations between tracks and detections. The first column corresponds
    to the null hypothesis (hence contains all ones).
likelihood_matrix: :class:`numpy.ndarray`
    A matrix of shape (num_tracks, num_detections + 1) containing the unnormalised
    likelihoods for all combinations of tracks and detections. The first column corresponds
    to the null hypothesis.

Returns
-------
:class:`numpy.ndarray`
    A matrix of shape (num_tracks, num_detections + 1) containing the normalised
    association probabilities for all combinations of tracks and detections. The first
    column corresponds to the null hypothesis.
'''

_CONSTRUCT_NOTE = (
    'Passing a validation matrix is deprecated and will no longer be possible in pyehm 3.0. A matrix that\n'
    'contains only 0s and 1s, or is passed as ``validation_matrix``, is interpreted as a pyehm 2.2 validation\n'
    'matrix. Pass ``log_likelihood_matrix`` by keyword to always interpret it as log-likelihoods.')

_COMPUTE_NOTE = (
    'Passing the matrix positionally or as ``likelihood_matrix`` is deprecated: it is interpreted as a pyehm 2.2\n'
    'likelihood matrix and probabilities are returned. From pyehm 3.0, a positional matrix is a log-likelihood\n'
    'matrix and log-probabilities are returned. Pass ``log_likelihood_matrix`` by keyword to use this now.')


def _with_deprecated_api(base, algorithm):
    """Subclass an algorithm with support for the pyehm 2.2 API, to be removed in pyehm 3.0."""

    def run(self, *args, **kwargs):
        if len(args) == 2 or 'validation_matrix' in kwargs or 'likelihood_matrix' in kwargs:
            validation_matrix, likelihood_matrix = bind(args, kwargs, ('validation_matrix', 'likelihood_matrix'))
            warn_deprecated('Calling run with a validation matrix and a likelihood matrix is deprecated; pass a '
                            'log-likelihood matrix instead, which returns log-probabilities.')
            return np.exp(base.run(self, to_log_likelihoods(validation_matrix, likelihood_matrix)))
        return base.run(self, *args, **kwargs)

    def construct(base_method):
        def method(self, *args, **kwargs):
            if 'validation_matrix' in kwargs or (len(args) == 1 and not kwargs and is_validation_matrix(args[0])):
                validation_matrix, = bind(args, kwargs, ('validation_matrix',))
                warn_deprecated(f'Passing a validation matrix to {base_method.__name__} is deprecated; pass a '
                                f'log-likelihood matrix instead, where -inf denotes invalid associations.')
                return base_method(self, to_log_likelihoods(validation_matrix))
            return base_method(self, *args, **kwargs)
        method.__name__ = base_method.__name__
        method.__doc__ = add_deprecation_note(base_method.__doc__, _CONSTRUCT_NOTE)
        return method

    def compute_association_probabilities(self, *args, **kwargs):
        compute = base.compute_association_probabilities
        if 'log_likelihood_matrix' in kwargs:
            return compute(self, *args, **kwargs)
        net, likelihood_matrix = bind(args, kwargs, ('net', 'likelihood_matrix'))
        warn_deprecated('Passing a likelihood matrix to compute_association_probabilities positionally or as '
                        'likelihood_matrix is deprecated; pass log_likelihood_matrix by keyword instead, which '
                        'returns log-probabilities. From pyehm 3.0, a positional matrix is a log-likelihood matrix.',
                        category=FutureWarning)
        with np.errstate(divide='ignore'):
            log_likelihood_matrix = np.log(np.asarray(likelihood_matrix, dtype=float))
        return np.exp(compute(self, net, log_likelihood_matrix))

    run.__doc__ = base.run.__doc__.rstrip('\n') + '\n' + _RUN_DEPRECATED_DOC.format(algorithm=algorithm)
    compute_association_probabilities.__doc__ = add_deprecation_note(
        base.compute_association_probabilities.__doc__, _COMPUTE_NOTE)

    return type(algorithm, (base,), {
        '__doc__': base.__doc__,
        '__module__': __name__,
        'run': run,
        'construct_net': construct(base.construct_net),
        'construct_tree': construct(base.construct_tree),
        'compute_association_probabilities': compute_association_probabilities,
    })


EHM = _with_deprecated_api(_EHM, 'EHM')
EHM2 = _with_deprecated_api(_EHM2, 'EHM2')
