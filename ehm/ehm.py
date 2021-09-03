import numpy as np
from .utils import EHMNetNode, EHM2NetNode, EHMNet, Tree, gen_clusters, calc_validation_and_likelihood_matrices


class EHM:

    @classmethod
    def run(cls, track_list, detection_list, hypotheses):
        """ Run EHM to compute and return association probabilities

        Parameters
        ----------
        track_list: list of :class:`Track`
            Current tracked objects
        detection_list : list of :class:`Detection`
            Retrieved measurements
        hypotheses: dict
            Key value pairs of tracks with associated detections

        Returns
        -------
        :class:`np.array`
            A matrix of shape (num_tracks, num_detections + 1) containing the normalised
            association probabilities for all combinations of tracks and detections. The first
            column corresponds to the null hypothesis.
        """

        # Get validation and likelihood matrices
        validation_matrix, likelihood_matrix = \
            calc_validation_and_likelihood_matrices(track_list, detection_list, hypotheses)

        # Cluster tracks into groups that share common detections
        clusters, missed_tracks = gen_clusters(validation_matrix)

        # Initialise the association probabilities matrix.
        assoc_prob_matrix = np.zeros(likelihood_matrix.shape)
        assoc_prob_matrix[missed_tracks, 0] = 1  # Null hypothesis is certain for missed tracks

        # Perform EHM for each cluster
        for cluster in clusters:

            # Extract track and detection indices
            # Note that the detection indices are adjusted to include the null hypothesis index (0)
            track_inds = np.sort(list(cluster.rows))
            detection_inds = np.sort(np.array(list(cluster.cols | {0})))

            # Extract validation and likelihood matrices for cluster
            c_validation_matrix = validation_matrix[track_inds, :][:, detection_inds]
            c_likelihood_matrix = likelihood_matrix[track_inds, :][:, detection_inds]

            # Construct the EHM net
            net = cls.construct_net(c_validation_matrix)

            # Compute the association probabilities
            c_assoc_prob_matrix = cls.compute_association_probabilities(net, c_likelihood_matrix)

            # Map the association probabilities to the main matrix
            for i, track_ind in enumerate(track_inds):
                assoc_prob_matrix[track_ind, detection_inds] = c_assoc_prob_matrix[i, :]

        return assoc_prob_matrix

    @staticmethod
    def construct_net(validation_matrix):
        """ Construct the EHM net as per Section 3.1 of [1]_

        Parameters
        ----------
        validation_matrix: :class:`np.array`
            An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
            (aka. valid) associations between tracks and detections. The first column corresponds
            to the null hypothesis (hence contains all ones).

        Returns
        -------
        : Net
            The constructed net object
        """
        num_tracks = validation_matrix.shape[0]

        # Initialise net
        root_node = EHMNetNode(layer=-1)  # Root node is at layer -1
        net = EHMNet([root_node])

        # A layer in the network is created for each track (not counting the root-node layer)
        num_layers = num_tracks
        for i in range(num_layers):

            # Get list of nodes in previous layer
            parent_nodes = [node for node in net.nodes if node.layer == i - 1]

            # Get indices of hypothesised detections for the track
            v_detections = set(np.flatnonzero(validation_matrix[i, :]))

            # For all nodes in previous layer
            for parent in parent_nodes:

                # Exclude any detections already considered by parent nodes (always include null)
                v_detections_m1 = (v_detections - parent.detections) | {0}

                # Iterate over valid detections
                for j in v_detections_m1:

                    # Get list of nodes in current layer
                    child_nodes = [node for node in net.nodes if node.layer == i]

                    # Compute remainders up to next layer (i+1)
                    remainders = set()
                    for ii in range(i + 1, num_layers):
                        remainders |= set(np.flatnonzero(validation_matrix[ii, :]))
                    remainders -= (parent.detections | {j}) - {0}

                    # Find valid nodes in current layer that have the same remainders
                    v_children = [child for child in child_nodes if remainders == child.remainders]

                    # If layer is empty or no valid nodes exist, add new node
                    if not len(v_children) or not len(child_nodes):
                        # Create new node
                        detections = parent.detections | {j}
                        child = EHMNetNode(layer=i, detections=detections, remainders=remainders)
                        # Add node to net
                        net.add_node(child, parent, j)
                    else:
                        # Simply add new edge or update existing one
                        for child in v_children:
                            if (parent, child) in net.edges:
                                net.edges[(parent, child)].add(j)
                            else:
                                net.edges[(parent, child)] = {j}
                            child.detections |= parent.detections | {j}
        return net

    @staticmethod
    def compute_association_probabilities(net, likelihood_matrix):
        """ Compute the joint association weights, as described in Section 3.3 of [1]_

        Parameters
        ----------
        net: Net
            A net object representing the valid joint association hypotheses
            (see :meth:`~.JPDAwithEHM._construct_ehm_net` and Section 3.1 of [1]_)
        likelihood_matrix: :class:`np.array`
            A matrix of shape (num_tracks, num_detections + 1) containing the unnormalised
            likelihoods for all combinations of tracks and detections. The first column corresponds
            to the null hypothesis.

        Returns
        -------
        :class:`np.array`
            A matrix of shape (num_tracks, num_detections + 1) containing the normalised
            association probabilities for all combinations of tracks and detecrtons. The first
            column corresponds to the null hypothesis.
        """
        num_tracks, num_detections = likelihood_matrix.shape
        num_nodes = net.num_nodes

        # Compute p_D (Downward-pass) - Eq. (22) of [1]
        p_D = np.zeros((num_nodes, ))
        p_D[0] = 1
        for child in net.nodes[1:]:
            c_i = child.ind
            parents = net.get_parents(child)
            for parent in parents:
                p_i = parent.ind
                ids = list(net.edges[(parent, child)])
                p_D[c_i] += np.sum(likelihood_matrix[child.layer, ids]*p_D[p_i])

        # Compute p_U (Upward-pass) - Eq. (23) of [1]
        p_U = np.zeros((num_nodes, ))
        p_U[-1] = 1
        for parent in reversed(net.nodes[:-1]):
            p_i = parent.ind
            children = net.get_children(parent)
            for child in children:
                c_i = child.ind
                ids = list(net.edges[(parent, child)])
                p_U[p_i] += np.sum(likelihood_matrix[child.layer, ids]*p_U[c_i])

        # Compute p_DT - Eq. (21) of [1]
        p_DT = np.zeros((num_detections, num_nodes))
        for child in net.nodes:
            c_i = child.ind
            v_edges = {edge: ids for edge, ids in net.edges.items() if edge[1] == child}
            for edge, ids in v_edges.items():
                p_i = edge[0].ind
                for j in ids:
                    p_DT[j, c_i] += p_D[p_i]

        # Compute p_T - Eq. (20) of [1]
        p_T = np.ones((num_detections, num_nodes))
        p_T[:, 0] = 0
        for node in net.nodes[1:]:
            n_i = node.ind
            for j in range(num_detections):
                p_T[j, n_i] = p_U[n_i]*likelihood_matrix[node.layer, j]*p_DT[j, n_i]

        # Compute association weights - Eq. (15) of [1]
        a_matrix = np.zeros(likelihood_matrix.shape)
        for i in range(num_tracks):
            node_inds = [n_i for n_i, node in enumerate(net.nodes) if node.layer == i]
            for j in range(num_detections):
                a_matrix[i, j] = np.sum(p_T[j, node_inds])
            # Normalise
            a_matrix[i, :] = a_matrix[i, :]/np.sum(a_matrix[i, :])

        return a_matrix


class EHM2:

    @classmethod
    def run(cls, track_list, detection_list, hypotheses):
        """ Run EHM to compute and return association probabilities

        Parameters
        ----------
        track_list: list of :class:`Track`
            Current tracked objects
        detection_list : list of :class:`Detection`
            Retrieved measurements
        hypotheses: dict
            Key value pairs of tracks with associated detections

        Returns
        -------
        :class:`np.array`
            A matrix of shape (num_tracks, num_detections + 1) containing the normalised
            association probabilities for all combinations of tracks and detections. The first
            column corresponds to the null hypothesis.
        """

        # Get validation and likelihood matrices
        validation_matrix, likelihood_matrix = \
            calc_validation_and_likelihood_matrices(track_list, detection_list, hypotheses)

        # Cluster tracks into groups that share common detections
        clusters, missed_tracks = gen_clusters(validation_matrix)

        # Initialise the association probabilities matrix.
        assoc_prob_matrix = np.zeros(likelihood_matrix.shape)
        assoc_prob_matrix[missed_tracks, 0] = 1  # Null hypothesis is certain for missed tracks

        # Perform EHM for each cluster
        for cluster in clusters:

            # Extract track and detection indices
            # Note that the detection indices are adjusted to include the null hypothesis index (0)
            track_inds = np.sort(list(cluster.rows))
            detection_inds = np.sort(np.array(list(cluster.cols | {0})))

            # Extract validation and likelihood matrices for cluster
            c_validation_matrix = validation_matrix[track_inds, :][:, detection_inds]
            c_likelihood_matrix = likelihood_matrix[track_inds, :][:, detection_inds]

            # Construct EHM2 tree
            tree = cls.construct_tree(c_validation_matrix)

            # Construct the EHM2 net
            net = cls.construct_net(c_validation_matrix)

            # Compute the association probabilities
            c_assoc_prob_matrix = cls.compute_association_probabilities(net, c_likelihood_matrix)

            # Map the association probabilities to the main matrix
            for i, track_ind in enumerate(track_inds):
                assoc_prob_matrix[track_ind, detection_inds] = c_assoc_prob_matrix[i, :]

        return assoc_prob_matrix

    @staticmethod
    def construct_tree(validation_matrix):
        """ Construct the EHM2 tree as per section 4.3 of [2]

        Parameters
        ----------
        validation_matrix: :class:`np.array`
            An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
            (aka. valid) associations between tracks and detections. The first column corresponds
            to the null hypothesis (hence contains all ones).

        Returns
        -------
        : Tree
            The constructed tree object

        """
        num_tracks = validation_matrix.shape[0]

        trees = []
        for i in reversed(range(num_tracks)):
            # Get indices of hypothesised detections for the track
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
                tree = Tree(i, children, detections)
                trees = [trees[j] for j in range(len(trees)) if j not in matched]
            else:
                children = []
                tree = Tree(i, children, v_detections)
            trees.append(tree)

        if len(trees) > 1:
            raise Exception

        return trees[0]

    @classmethod
    def construct_net(cls, validation_matrix, tree):
        """ Construct the EHM net as per Section 3.1 of [1]_

        Parameters
        ----------
        validation_matrix: :class:`np.array`
            An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
            (aka. valid) associations between tracks and detections. The first column corresponds
            to the null hypothesis (hence contains all ones).

        Returns
        -------
        : Net
            The constructed net object
        """
        num_tracks = validation_matrix.shape[0]

        # Initialise net
        subnet = 0
        root_node = EHM2NetNode(layer=-1, subnet=subnet, track_ind=-1)  # Root node is at layer -1
        net = EHMNet([root_node])

        # A layer in the network is created for each track (not counting the root-node layer)
        cls._construct_net(net, validation_matrix, tree, -1, 0)

        for i in range(num_tracks):
            net.nodes_per_track[i] = [node for node in net.nodes if node.track_ind==i]
        return net

    @classmethod
    def _construct_net(cls, net, validation_matrix, tree, layer, subnet):
        layer += 1
        track_ind = tree.track

        # Get list of nodes in previous layer
        parent_nodes = [node for node in net.nodes if node.layer == layer - 1 and node.subnet == subnet]

        # Get indices of hypothesised detections for the track
        v_detections = set(np.flatnonzero(validation_matrix[track_ind, :]))

        # For all nodes in previous layer
        for parent in parent_nodes:

            # Exclude any detections already considered by parent nodes (always include null)
            v_detections_m1 = (v_detections - parent.detections) | {0}

            if tree.children:
                for i, sub_tree in enumerate(tree.children):

                    # Iterate over valid detections
                    for j in v_detections_m1:

                        # Get list of nodes in current layer
                        child_nodes = [node for node in net.nodes if node.layer == layer and node.subnet == subnet+i]

                        # Compute remainders up to next layer (i+1)
                        remainders = {0}
                        remainders |= sub_tree.detections
                        remainders -= (parent.detections | {j}) - {0}

                        # Find valid nodes in current layer that have the same remainders
                        v_children = [child for child in child_nodes if remainders == child.remainders]

                        # If layer is empty or no valid nodes exist, add new node
                        if not len(v_children) or not len(child_nodes):
                            # Create new node
                            detections = parent.detections | {j}
                            child = EHM2NetNode(layer=layer, subnet=subnet+i, track_ind=track_ind, detections=detections,
                                                remainders=remainders)
                            # Add node to net
                            net.add_node(child, parent, j)
                        else:
                            # Simply add new edge or update existing one
                            for child in v_children:
                                if (parent, child) in net.edges:
                                    net.edges[(parent, child)].add(j)
                                else:
                                    net.edges[(parent, child)] = {j}
                                child.detections |= parent.detections | {j}
            else:
                child = next((node for node in net.nodes if node.layer == layer and node.subnet == subnet), None)
                # Iterate over valid detections
                for j in v_detections_m1:

                    # If layer is empty or no valid nodes exist, add new node
                    if not child:
                        # Create new node
                        detections = parent.detections | {j}
                        child = EHM2NetNode(layer=layer, subnet=subnet, track_ind=track_ind, detections=detections,
                                            remainders={})
                        # Add node to net
                        net.add_node(child, parent, j)
                    else:
                        # Simply add new edge or update existing one
                        if (parent, child) in net.edges:
                            net.edges[(parent, child)].add(j)
                        else:
                            net.edges[(parent, child)] = {j}
                        child.detections |= parent.detections | {j}

        for i, sub_tree in enumerate(tree.children):
            cls._construct_net(net, validation_matrix, sub_tree, layer, subnet + i)

    @classmethod
    def compute_association_probabilities(cls, net, tree, likelihood_matrix):
        """ Compute the joint association weights, as described in Section 3.3 of [1]_

        Parameters
        ----------
        net: Net
            A net object representing the valid joint association hypotheses
            (see :meth:`~.JPDAwithEHM._construct_ehm_net` and Section 3.1 of [1]_)
        likelihood_matrix: :class:`np.array`
            A matrix of shape (num_tracks, num_detections + 1) containing the unnormalised
            likelihoods for all combinations of tracks and detections. The first column corresponds
            to the null hypothesis.

        Returns
        -------
        :class:`np.array`
            A matrix of shape (num_tracks, num_detections + 1) containing the normalised
            association probabilities for all combinations of tracks and detecrtons. The first
            column corresponds to the null hypothesis.
        """
        num_tracks, num_detections = likelihood_matrix.shape
        num_nodes = net.num_nodes

        nodes_backwards = cls.order_backwards(net, tree)
        nodes_forwards = list(reversed(nodes_backwards))

        # Compute w_B (Backward-pass) - Eq. (47) of [1]
        w_B = np.zeros((num_nodes,))
        for parent in nodes_backwards:
            p_i = parent.ind
            if not parent.remainders and parent.layer>-1:
                w_B[p_i] = 1
                continue
            w = dict()
            children = net.get_children(parent)
            for child in children:
                c_i = child.ind
                det_inds = list(net.edges[(parent, child)])
                for det_ind in det_inds:
                    if det_ind in w:
                        w[det_ind] = w[det_ind]*w_B[c_i]
                    else:
                        w[det_ind] = likelihood_matrix[child.track_ind, det_ind]*w_B[c_i]
                a = 2
            w_B[p_i] = np.sum([wi for _, wi in w.items()])

        # Compute w_F (Forward-pass) - Eq. (49) of [1]
        w_F = np.zeros((num_nodes,))
        w_F[0] = 1
        for child in nodes_forwards[1:]:
            c_i = child.ind
            w = dict()
            parents = net.get_children(child)
            for parent in parents:
                p_i = parent.ind
                det_inds = list(net.edges[(parent, child)])
                for det_ind in det_inds:
                    




    @classmethod
    def order_backwards(cls, net, tree):
        order = [net.nodes[0]]  # The root node
        return cls.order_backwards_recursive(net, tree) + order

    @classmethod
    def order_backwards_recursive(cls, net, tree):
        order = net.nodes_per_track[tree.track]
        for child in tree.children:
            order_n = cls.order_backwards_recursive(net, child)
            order = order_n + order
        return order