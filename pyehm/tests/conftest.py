import numpy as np
import pytest


@pytest.fixture()
def scenarios():
    inputs = dict()
    inputs['scenarioA'] = dict()
    inputs['scenarioA']['validation_matrix'] = np.array([[1, 1, 0, 0, 0],
                                                         [1, 1, 1, 1, 0],
                                                         [1, 1, 1, 0, 0],
                                                         [1, 0, 0, 1, 1]])
    inputs['scenarioA']['likelihood_matrix'] = np.array([[0.1, 0.9, 0, 0, 0],
                                                         [0.1, 0.3, 0.2, 0.4, 0],
                                                         [0.7, 0.1, 0.2, 0, 0],
                                                         [0.2, 0, 0, 0.75, 0.05]])

    inputs['scenarioB'] = dict()
    inputs['scenarioB']['validation_matrix'] = np.array([[1, 1, 0, 1, 0, 0, 0, 0],
                                                         [1, 0, 1, 0, 0, 1, 0, 0],
                                                         [1, 1, 0, 0, 0, 0, 0, 1],
                                                         [1, 0, 0, 0, 1, 0, 0, 1],
                                                         [1, 0, 1, 0, 0, 0, 0, 0],
                                                         [1, 0, 0, 0, 0, 0, 0, 1],
                                                         [1, 0, 1, 0, 0, 0, 1, 0],
                                                         [1, 0, 1, 0, 1, 0, 0, 0],
                                                         [1, 0, 0, 0, 1, 0, 0, 0]])
    inputs['scenarioB']['likelihood_matrix'] = np.array([[0.1, 0.9, 0, 0.1, 0, 0, 0, 0],
                                                         [0.6, 0, 0.2, 0, 0, 0.2, 0, 0],
                                                         [0.2, 0.6, 0, 0, 0, 0, 0, 0.2],
                                                         [0.1, 0, 0, 0, 0.45, 0, 0, 0.45],
                                                         [0.2, 0, 0.8, 0, 0, 0, 0, 0],
                                                         [0.3, 0, 0, 0, 0, 0, 0, 0.7],
                                                         [0.25, 0, 0.35, 0, 0, 0, 0.4, 0],
                                                         [0.2, 0, 0.5, 0, 0.3, 0, 0, 0],
                                                         [0.8, 0, 0, 0, 0.2, 0, 0, 0]])

    inputs['scenarioC'] = dict()
    inputs['scenarioC']['validation_matrix'] = np.array([[1, 1, 0, 1, 0, 0, 0, 0],
                                                         [1, 0, 1, 0, 0, 1, 0, 0],
                                                         [1, 1, 0, 0, 0, 0, 0, 1],
                                                         [1, 0, 0, 0, 1, 0, 0, 1],
                                                         [1, 0, 1, 0, 0, 0, 0, 0],
                                                         [1, 0, 0, 0, 0, 0, 0, 0],
                                                         [1, 0, 0, 0, 0, 0, 1, 0],
                                                         [1, 0, 0, 0, 1, 0, 0, 0],
                                                         [1, 0, 0, 0, 0, 0, 0, 0]])

    inputs['scenarioC']['likelihood_matrix'] = np.array([[0.1, 0.9, 0, 0.1, 0, 0, 0, 0],
                                                         [0.6, 0, 0.2, 0, 0, 0.2, 0, 0],
                                                         [0.2, 0.6, 0, 0, 0, 0, 0, 0.2],
                                                         [0.1, 0, 0, 0, 0.45, 0, 0, 0.45],
                                                         [0.2, 0, 0.8, 0, 0, 0, 0, 0],
                                                         [1, 0, 0, 0, 0, 0, 0, 0.0],
                                                         [0.55, 0, 0, 0, 0, 0, 0.45, 0],
                                                         [0.2, 0, 0, 0, 0.8, 0, 0, 0],
                                                         [1, 0, 0, 0, 0, 0, 0, 0]])
    return inputs
