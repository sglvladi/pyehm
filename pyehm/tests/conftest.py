import numpy as np
import pytest


def pytest_configure(config):
    # Tests must not use deprecated pyehm APIs, except where they check them with pytest.deprecated_call
    config.addinivalue_line('filterwarnings', 'error::DeprecationWarning:pyehm')
    config.addinivalue_line('filterwarnings', 'error::FutureWarning:pyehm')


@pytest.fixture()
def scenarios():
    inputs = dict()
    inputs['scenarioA'] = dict()
    inputs['scenarioA']['likelihood_matrix'] = np.array([[0.1, 0.9, 0, 0, 0],
                                                         [0.1, 0.3, 0.2, 0.4, 0],
                                                         [0.7, 0.1, 0.2, 0, 0],
                                                         [0.2, 0, 0, 0.75, 0.05]])

    inputs['scenarioB'] = dict()
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
