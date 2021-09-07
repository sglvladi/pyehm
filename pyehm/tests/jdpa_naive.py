import itertools
import numpy as np


def jpda_naive(validation_matrix, likelihood_matrix):
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
        for hyp in joint_hyp:
            track = hyp[0]
            detection = hyp[1]
            lik *= likelihood_matrix[track, detection]
        valid_joint_hypotheses_lik[joint_hyp] = lik

    assoc_matrix = np.zeros((num_tracks, num_detections))
    for track in range(num_tracks):
        v_detections = np.flatnonzero(validation_matrix[track, :])
        for detection in v_detections:
            prob = np.sum([lik for hyp, lik in valid_joint_hypotheses_lik.items() if (track, detection) in hyp])
            assoc_matrix[track, detection] = prob
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


def gen_random_scenario(num_tracks, num_detections):

    validation_matrix = np.zeros((num_tracks, num_detections+1))
    validation_matrix[:, 0] = 1
    for i in range(num_tracks):
        validation_matrix[i, 1:] = np.random.randint(0, 2, (num_detections,))

    likelihood_matrix = np.zeros((num_tracks, num_detections + 1))
    for i in range(num_tracks):
        liks = np.random.rand(num_detections+1)
        likelihood_matrix[i, :] = validation_matrix[i, :]*liks

    return validation_matrix, likelihood_matrix
