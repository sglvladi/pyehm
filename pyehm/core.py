import numpy as np

from .utils import EHMNetNode, EHMNet, EHM2Tree, gen_clusters, compute_identity


class EHM:
    """ Efficient Hypothesis Management 2 (EHM2)

    An implementation of the EHM2 algorithm, as documented in [EHM2]_.
    """

    @classmethod
    def run(cls, validation_matrix, likelihood_matrix):
        """Run EHM to compute and return association probabilities

        Parameters
        ----------
        validation_matrix : :class:`numpy.ndarray`
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
        """

        # Cluster tracks into groups that share common detections
        clusters, missed_tracks = gen_clusters(validation_matrix, likelihood_matrix)

        # Initialise the association probabilities matrix.
        assoc_prob_matrix = np.zeros(likelihood_matrix.shape)
        assoc_prob_matrix[missed_tracks, 0] = 1  # Null hypothesis is certain for missed tracks

        # Perform EHM for each cluster
        for cluster in clusters:

            # Extract track and detection indices
            c_tracks = cluster.tracks
            c_detections = cluster.detections

            # Extract validation and likelihood matrices for cluster
            c_validation_matrix = cluster.validation_matrix
            c_likelihood_matrix = cluster.likelihood_matrix

            # Construct the EHM net
            net = cls.construct_net(c_validation_matrix)

            # Compute the association probabilities
            c_assoc_prob_matrix = cls.compute_association_probabilities(net, c_likelihood_matrix)

            # Map the association probabilities to the main matrix
            for i, track in enumerate(c_tracks):
                assoc_prob_matrix[track, c_detections] = c_assoc_prob_matrix[i, :]

        return assoc_prob_matrix

    @classmethod
    def construct_net(cls, validation_matrix):
        """ Construct the EHM net as per Section 4 of [EHM2]_

        Parameters
        ----------
        validation_matrix: :class:`numpy.ndarray`
            An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
            (aka. valid) associations between tracks and detections. The first column corresponds
            to the null hypothesis (hence contains all ones).

        Returns
        -------
        : :class:`~.EHMNet`
            The constructed net object

        Raises
        ------
        ValueError
            If the provided ``validation_matrix`` is such that tracks can be divided into separate clusters. See
            the :ref:`Note <note1>` below for work-around.


        .. _note1:

        .. note::
            If the provided ``validation_matrix`` is such that tracks can be divided into separate clusters, this
            method will raise a ValueError exception. To work-around this issue, you can use the
            :func:`~pyehm.utils.gen_clusters` function to first generate individual clusters and then generate a net
            for each cluster, as shown below:

            .. code-block:: python

                from pyehm.core import EHM2
                from pyehm.utils import gen_clusters

                validation_matrix = <Your validation matrix>

                clusters, _ = gen_clusters(validation_matrix)

                nets = []
                for cluster in clusters:
                    nets.append(EHM2.construct_net(cluster.validation_matrix)

        """
        num_tracks = validation_matrix.shape[0]

        # Construct tree
        try:
            tree = cls.construct_tree(validation_matrix)
        except ValueError:
            raise ValueError('The provided validation matrix results in multiple clusters of tracks')

        # Initialise net
        root_node = EHMNetNode(layer=0)
        net = EHMNet([root_node], validation_matrix=validation_matrix, tree=tree)

        # Recursively construct next layers
        cls._construct_net_layer(net, tree)

        return net

    @classmethod
    def _construct_net_layer(cls, net, tree):

        # Get list of nodes in previous layer of subtree
        try:
            parent_nodes_ind = net.get_nodes_by_layer(tree.track)
            parent_nodes = net.get_nodes(parent_nodes_ind)
        except KeyError:
            parent_nodes = set()

        # Get indices of hypothesised detections for the track
        v_detections = set(np.flatnonzero(net.validation_matrix[tree.track, :]))

        # If this is not an end layer
        if tree.children:

            # Process each subtree
            for child_tree in tree.children:

                # Compute accumulated measurements up to next layer (i+1)
                acc = child_tree.detections

                # List of nodes in current layer
                children_per_identity = dict()

                # For all nodes in previous layer
                for parent in parent_nodes:

                    # Exclude any detections already considered by parent nodes (always include null)
                    v_detections_m1 = (v_detections - parent.identity) | {0}

                    # Iterate over valid detections
                    for j in v_detections_m1:

                        # Identity
                        # identity = acc.intersection(parent.identity | {j}) - {0}
                        identity = compute_identity(acc, parent.identity, j)

                        # Find valid nodes in current layer that have the same identity
                        try:
                            v_children_inds = children_per_identity[tuple(sorted(identity))]
                            v_children = net.get_nodes(v_children_inds)
                        except KeyError:
                            v_children = set()

                        # If layer is empty or no valid nodes exist, add new node
                        if not len(v_children):
                            # Create new node
                            child = EHMNetNode(layer=child_tree.track, identity=identity)
                            # Add node to net
                            net.add_node(child, parent, j)
                            # Add node to list of child nodes
                            try:
                                children_per_identity[tuple(sorted(child.identity))].add(child.ind)
                            except KeyError:
                                children_per_identity[tuple(sorted(child.identity))] = {child.ind}
                        else:
                            # Simply add new edge or update existing one
                            for child in v_children:
                                net.add_edge(parent, child, j)
        else:
            # For all nodes in previous layer
            for parent in parent_nodes:

                # Exclude any detections already considered by parent nodes (always include null)
                v_detections_m1 = (v_detections - parent.identity) | {0}

                # Get leaf child, if any
                try:
                    child = net.nodes[next(iter(net.get_nodes_by_layer(-1)))]
                except (KeyError, StopIteration):
                    child = None

                # Iterate over valid detections
                for j in v_detections_m1:

                    # If layer is empty or no valid node exist, add new node
                    if not child:
                        # Create new node
                        child = EHMNetNode(layer=-1)
                        # Add node to net
                        net.add_node(child, parent, j)
                    else:
                        # Simply add new edge or update existing one
                        net.add_edge(parent, child, j)

        # Create new layers for each sub-tree
        for i, child_tree in enumerate(tree.children):
            cls._construct_net_layer(net, child_tree)

    @staticmethod
    def compute_association_probabilities(net, likelihood_matrix):
        """ Compute the joint association weights, as described in Section 4.2 of [EHM2]_

        Parameters
        ----------
        net: :class:`~.EHMNet`
            A net object representing the valid joint association hypotheses
        likelihood_matrix: :class:`numpy.ndarray`
            A matrix of shape (num_tracks, num_detections + 1) containing the unnormalised
            likelihoods for all combinations of tracks and detections. The first column corresponds
            to the null hypothesis.

        Returns
        -------
        :class:`numpy.ndarray`
            A matrix of shape (num_tracks, num_detections + 1) containing the normalised
            association probabilities for all combinations of tracks and detecrtons. The first
            column corresponds to the null hypothesis.
        """
        num_tracks, num_detections = likelihood_matrix.shape
        num_nodes = net.num_nodes

        nodes_forwards = net.nodes_forward
        nodes_backwards = list(reversed(nodes_forwards))

        # Precompute valid detections per track
        v_detections_per_track = [set(np.flatnonzero(row)) for row in net.validation_matrix]

        # Compute w_B (Backward-pass) - Eq. (47) of [EHM2]
        w_B = np.zeros((num_nodes,))
        for parent in nodes_backwards:
            p_i = parent.ind

            # If parent is a leaf node
            if parent.layer < 0:
                w_B[p_i] = 1
                continue

            child_layers = net.child_layers[parent.layer]
            v_detections = v_detections_per_track[parent.layer] - parent.identity
            weights_per_det = {det_ind: likelihood_matrix[parent.layer, det_ind]
                               for det_ind in v_detections}

            for det_ind in weights_per_det:
                # for child_layer in child_layers:
                #     identity = net.acc_per_layer[child_layer].intersection(parent.identity | {det_ind}) - {0}
                #     v_children = [c_i for c_i in net.nodes_per_layer[child_layer] if net.nodes[c_i].identity == identity]
                #     weights_per_det[det_ind] *= np.prod([w_B[c_i] for c_i in v_children])
                v_children = net.get_children_per_detection(parent, det_ind)
                # v_children = net.children_per_detection.get((p_i, det_ind), [])
                weights_per_det[det_ind] *= np.prod([w_B[c_i] for c_i in v_children])
            w_B[p_i] = np.sum([w for w in weights_per_det.values()])

        # Compute w_F (Forward-pass) - Eq. (49) of [EHM2]
        w_F = np.zeros((num_nodes,))
        w_F[0] = 1
        for parent in nodes_forwards:
            # Skip the leaf nodes
            if parent.layer < 0:
                continue
            p_i = parent.ind
            v_detections = v_detections_per_track[parent.layer] - parent.identity
            a=2
            for det_ind in v_detections:
                # v_children = net.children_per_detection.get((p_i, det_ind), [])
                v_children = net.get_children_per_detection(parent, det_ind)
                for c_i in v_children:
                    child = net.nodes[c_i]
                    if child.layer < 0:
                        continue
                    sibling_inds = list(filter(lambda x: x != c_i, v_children))
                    sibling_weight = np.prod(w_B[sibling_inds]) if len(sibling_inds) > 0 else 1
                    weight = likelihood_matrix[parent.layer, det_ind] * w_F[p_i] * sibling_weight
                    w_F[c_i] += weight

        # Compute association probs - Eq. (46) of [EHM2]
        a_matrix = np.zeros(likelihood_matrix.shape)
        for track in range(num_tracks):
            v_detections = v_detections_per_track[track]
            for parent in net.get_nodes(net.get_nodes_by_layer(track)):
                v_detections_tmp = v_detections - parent.identity
                for detection in v_detections_tmp:
                    # v_children = net.children_per_detection.get((parent.ind, detection), [])
                    v_children = net.get_children_per_detection(parent, detection)
                    if not v_children:
                        continue
                    weight = likelihood_matrix[track, detection] * w_F[parent.ind]
                    for c_i in v_children:
                        weight *= w_B[c_i]
                    a_matrix[track, detection] += weight
            a_matrix[track, :] /= np.sum(a_matrix[track, :])

        return a_matrix

    @staticmethod
    def construct_tree(validation_matrix):
        """ Construct the EHM2 tree as per section 4.3 of [EHM2]_

        Parameters
        ----------
        validation_matrix: :class:`numpy.ndarray`
            An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
            (aka. valid) associations between tracks and detections. The first column corresponds
            to the null hypothesis (hence contains all ones).

        Returns
        -------
        : :class:`~.EHM2Tree`
            The constructed tree object

        Raises
        ------
        ValueError
            If the provided ``validation_matrix`` is such that tracks can be divided into separate clusters. See
            the :ref:`Note <note2>` below for work-around.


        .. _note2:

        .. note::
            If the provided ``validation_matrix`` is such that tracks can be divided into separate clusters, this
            method will raise a ValueError exception. To work-around this issue, you can use the
            :func:`~pyehm.utils.gen_clusters` function to first generate individual clusters and then generate a tree
            for each cluster, as shown below:

            .. code-block:: python

                from pyehm.core import EHM2
                from pyehm.utils import gen_clusters

                validation_matrix = <Your validation matrix>

                clusters, _ = gen_clusters(validation_matrix)

                trees = []
                for cluster in clusters:
                    trees.append(EHM2.construct_tree(cluster.validation_matrix)

        """
        num_tracks = validation_matrix.shape[0]

        tree = None
        for i in reversed(range(num_tracks)):
            # Get indices of hypothesised detections for the track (minus the null hypothesis)
            v_detections = set(np.flatnonzero(validation_matrix[i, :])) - {0}

            if tree is None:
                children = []
                tree = EHM2Tree(i, children, v_detections)
            else:
                detections = set()
                detections |= tree.detections
                detections |= v_detections
                tree = EHM2Tree(i, [tree], detections)

        for node in tree.nodes:
            node.detections = frozenset(node.detections)

        return tree


class EHM2(EHM):
    @staticmethod
    def construct_tree(validation_matrix):
        """ Construct the EHM2 tree as per section 4.3 of [EHM2]_

        Parameters
        ----------
        validation_matrix: :class:`numpy.ndarray`
            An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
            (aka. valid) associations between tracks and detections. The first column corresponds
            to the null hypothesis (hence contains all ones).

        Returns
        -------
        : :class:`~.EHM2Tree`
            The constructed tree object

        Raises
        ------
        ValueError
            If the provided ``validation_matrix`` is such that tracks can be divided into separate clusters. See
            the :ref:`Note <note2>` below for work-around.


        .. _note2:

        .. note::
            If the provided ``validation_matrix`` is such that tracks can be divided into separate clusters, this
            method will raise a ValueError exception. To work-around this issue, you can use the
            :func:`~pyehm.utils.gen_clusters` function to first generate individual clusters and then generate a tree
            for each cluster, as shown below:

            .. code-block:: python

                from pyehm.core import EHM2
                from pyehm.utils import gen_clusters

                validation_matrix = <Your validation matrix>

                clusters, _ = gen_clusters(validation_matrix)

                trees = []
                for cluster in clusters:
                    trees.append(EHM2.construct_tree(cluster.validation_matrix)

        """
        num_tracks = validation_matrix.shape[0]

        trees = []
        for i in reversed(range(num_tracks)):
            # Get indices of hypothesised detections for the track (minus the null hypothesis)
            v_detections = set(np.flatnonzero(validation_matrix[i, :])) - {0}

            matched = []
            for j, tree in enumerate(trees):
                if v_detections.intersection(tree.detections):
                    matched.append(j)

            if matched:
                children = [trees[j] for j in matched]
                detections = set()
                for tree in children:
                    detections |= tree.detections
                detections |= v_detections
                tree = EHM2Tree(i, children, detections)
                trees = [trees[j] for j in range(len(trees)) if j not in matched]
            else:
                children = []
                tree = EHM2Tree(i, children, v_detections)
            trees.append(tree)

        if len(trees) > 1:
            raise ValueError('The provided validation matrix results in multiple clusters of tracks')

        tree = trees[0]

        for node in tree.nodes:
            node.detections = frozenset(node.detections)

        return tree
