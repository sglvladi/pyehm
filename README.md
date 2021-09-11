# Python Efficient Hypothesis Management (PyEHM)

[![License](https://img.shields.io/badge/License-EPL%202.0-red.svg)](https://opensource.org/licenses/EPL-2.0)
[![CircleCI](https://circleci.com/gh/sglvladi/pyehm/tree/main.svg?style=svg&circle-token=4c2f3fb99ef265443b75ccdc5bc7bf118582a4db)](https://circleci.com/gh/sglvladi/pyehm/tree/main)

PyEHM is a Python package that includes open-souce implementations of the Efficient Hypothesis Management (EHM) 
Algorithms described in [1], [2] and **covered by the patent** [3].

> [1] Maskell, S., Briers, M. and Wright, R., 2004, August. Fast mutual exclusion. In Signal and Data Processing of 
Small Targets 2004 (Vol. 5428, pp. 526-536). International Society for Optics and Photonics
> 
> [2] Horridge, P. and Maskell, S., 2006, July. Real-time tracking of hundreds of targets with efficient exact JPDAF 
implementation. In 2006 9th International Conference on Information Fusion (pp. 1-8). IEEE 
> 
> [3] Maskell, S., 2003, July. Signal Processing with Reduced Combinatorial Complexity. Patent Reference:0315349.1

## Installation
PyEHM is currently in active development under *beta*. To install the latest version from the GitHub repository:

```
python -m pip install git+https://github.com/sglvladi/pyehm#egg=pyehm
```

### Developing
If you are looking to carry out development with PyEHM, you should first clone from GitHub and install with development
dependencies by doing the following:

```
git clone "https://github.com/sglvladi/pyehm"
cd pyehm
python -m pip install -e .[dev]
```

## Usage
### Standalone
The core components in PyEHM are the ```EHM``` and ```EHM2``` classes, which constitute implementations of the EHM [1]
and EHM2 [2] algorithms for data association. 

#### Formulating the possible associations between targets and measurements
Both classes operate on a ```validation_matrix``` and a ```likelihood_matrix```. The ```validation_matrix``` is a matrix 
that represents the possible associations between different targets and measurements, while the ```likelihood_matrix```
contains the respective likelihoods/probabilities of these associations. Both matrices have a shape ```(N_T, N_M+1)```,
where ```N_T``` is the number of targets and ```N_M``` is the numer of measurements. 

For example, assume we have the following scenario of 4 targets and 4 measurements (taken from Section 4.4 of [2]):

| Target index   | Gated measurement indices |
| -------------- | ------------------------- |
| 0              | 0, 1                      |
| 1              | 0, 1, 2, 3                |
| 2              | 0, 1, 2                   |
| 3              | 0, 3, 4                   |

where the null measurement hypothesis is given the index of 0. Then the ```validation_matrix``` would be a ```(4, 5)```
numpy array of the following form:

```python
validation_matrix = np.array([[1, 1, 0, 0, 0],  # 0 -> 0,1
                              [1, 1, 1, 1, 0],  # 1 -> 0,1,2,3
                              [1, 1, 1, 0, 0],  # 2 -> 0,1,2
                              [1, 0, 0, 1, 1]]) # 3 -> 0,3,4
```
The ```likelihood_matrix``` is such that each element ```likelihood_matrix[i, j]``` contains the respective likelihood 
of target ```i``` being associated to measurement ```j```. Therefore, based on the above example, the ```likelihood_matrix```
could be the following:

```python
likelihood_matrix = np.array([[0.1, 0.9, 0, 0, 0],
                              [0.1, 0.3, 0.2, 0.4, 0],
                              [0.7, 0.1, 0.2, 0, 0],
                              [0.2, 0, 0, 0.75, 0.05]])
```

#### Computing joint association probabilities
Based on the above, we can use either ```EHM``` or ```EHM2``` to compute the joint association probabilities matrix
```assoc_matrix``` as follows:
```python
from pyehm.core import EHM, EHM2

# Using EHM
assoc_matrix = EHM.run(validation_matrix, likelihood_matrix)

# Using EHM2
assoc_matrix = EHM2.run(validation_matrix, likelihood_matrix)
```
Note that both ```EHM``` and ```EHM2``` should produce the same results, although ```EHM2``` should, in principle, be 
faster.

### Stone Soup plugin
PyEHM includes implementations of [Stone Soup](https://stonesoup.readthedocs.io/en/v0.1b6/index.html) compatible Joint 
Probabilistic Data Association (JPDA) [DataAssociator](https://stonesoup.readthedocs.io/en/v0.1b6/stonesoup.dataassociator.html)
classes. 

These are provided under the [```JPDAWithEHM```](https://github.com/sglvladi/pyehm/blob/main/pyehm/jpda.py#L12) 
and [```JPDAWithEHM2```](https://github.com/sglvladi/pyehm/blob/main/pyehm/jpda.py#L108) classes, which implement the 
EHM and EHM2 algorithms, respectively.


Assuming that both Stone Soup and PyEHM are installed in your Python environment, the ```JPDAWithEHM``` and 
```JPDAWithEHM2``` classes can be used as drop-in replacements to the standard Stone Soup
[```JPDA```](https://stonesoup.readthedocs.io/en/v0.1b6/stonesoup.dataassociator.html#stonesoup.dataassociator.probability.JPDA)
data associator as follows:

```python
from stonesoup.plugins.pyehm import JPDAWithEHM, JPDAWithEHM2

associator = JPDAWithEHM(hypothesiser)
# Or
associator = JPDAWithEHM2(hypothesiser)
```

## License
PyEHM is licenced under Eclipse Public License 2.0. See [License](https://github.com/sglvladi/pyehm/blob/main/LICENSE.md) for 
more details.

This software is the property of [QinetiQ Limited](https://www.qinetiq.com/en/) and any requests for use of the
software for commercial use or other use outside of the Eclipse Public Licence should be made to QinetiQ
Limited.