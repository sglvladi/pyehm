#include "Docstrings.h"
namespace docstrings {

    std::string EHMNetNode() {
        return R"(EHMNetNode(layer: int, identity: Set[int])
A node in the :class:`~.EHMNet` constructed by :class:`~.EHM`.

Parameters
----------
layer: :class:`int`
    Index of the network layer in which the node is placed. Since a different layer in the network is built for
    each track, this also represented the index of the track this node relates to.
identity: :class:`set` of :class:`int`
    The identity of the node. As per Section 3.1 of [EHM1]_, "the identity for each node is an indication of how
    measurement assignments made for tracks already considered affect assignments for tracks remaining to be
    considered".
)";
    }

    std::string EHM2NetNode() {
        return R"(EHM2NetNode(layer: int, track: int, subnet: int, identity: Set[int])
A node in the :class:`~.EHM2Net` constructed by :class:`~.EHM2`.

Parameters
----------
layer: :class:`int`
    Index of the network layer in which the node is placed.
track: :class:`int`
    Index of track this node relates to.
subnet: :class:`int`
    Index of subnet to which the node belongs.
identity: :class:`set` of :class:`int`
    The identity of the node. As per Section 3.1 of [EHM1]_, "the identity for each node is an indication of how
    measurement assignments made for tracks already considered affect assignments for tracks remaining to be
    considered".
)";
    }

    std::string EHMNet()
    {
        return R"(EHMNet(root: EHMNetNode, validation_matrix: numpy.ndarray)
Represents the nets constructed by :class:`~.EHM`.

Parameters
----------
root: :class:`~.EHMNetNode`
    The net root node.
validation_matrix: :class:`numpy.ndarray`
    An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
    (aka. valid) associations between tracks and detections. The first column corresponds
    to the null hypothesis (hence contains all ones).
)";
    }

    std::string EHMNet___init__()
    {
        return R"(__init__(root: pyehm.net.EHMNetNode, validation_matrix: numpy.ndarray)";
    }

    std::string EHMNet_num_layers()
    {
        return R"(num_layers() -> int
Number of layers in the net)";
    }

    std::string EHMNet_num_nodes()
    {
        return R"(Number of nodes in the net)";
    }

    std::string EHMNet_root()
    {
        return R"(The root node of the net)";
    }

    std::string EHMNet_nodes()
    {
        return R"(The nodes comprising the net)";
    }

    std::string EHMNet_nodes_forward()
    {
        return R"(The net nodes, ordered by increasing layer)";
    }

    std::string EHMNet_get_parents()
    {
        return R"(get_parents(node: EHMNetNode) -> List[EHMNetNode]
Get the parents of a node.

Parameters
----------
node: :class:`~.EHMNetNode`
    The node whose parents should be returned

Returns
-------
:class:`list` of :class:`~.EHMNetNode`
    List of parent nodes
)";
    }

    std::string EHMNet_get_children()
    {
        return R"(get_children(node: EHMNetNode) -> List[EHMNetNode]
Get the children of a node.

Parameters
----------
node: :class:`~.EHMNetNode`
    The node whose children should be returned

Returns
-------
:class:`list` of :class:`~.EHMNetNode`
    List of child nodes
)";
    }

    std::string EHMNet_get_edges()
    {
        return R"(get_edges(parent: EHMNetNode, child: EHMNetNode) -> List[int]
Get edges between two nodes.

Parameters
----------
parent: :class:`~.EHMNetNode`
    The parent node, i.e. the source of the edge.
child: :class:`~.EHMNetNode`
    The child node, i.e. the target of the edge.

Returns
-------
:class:`list` of :class:`int`
    Indices of measurements representing the parent child relationship.
)";
    }

    std::string EHMNet_add_node()
    {
        return R"(add_node(node: EHMNetNode, parent: EHMNetNode, detection: int)
Add a node to the network.

Parameters
----------
node: :class:`~.EHMNetNode`
    The node to be added.
parent: :class:`~.EHMNetNode`
    The parent of the node.
detection: :class:`int`
    Index of measurement representing the parent child relationship.
)";
    }

    std::string EHMNet_add_edge()
    {
        return R"(add_edge(parent: EHMNetNode, child: EHMNetNode, detection: int)
Add edge between two nodes, or update an already existing edge by adding the detection to it.

Parameters
----------
parent: :class:`~.EHMNetNode`
    The parent node, i.e. the source of the edge.
child: :class:`~.EHMNetNode`
    The child node, i.e. the target of the edge.
detection: :class:`int`
    Index of measurement representing the parent child relationship.
)";
    }

    std::string EHM2Net()
    {
        return R"(EHM2Net(root: EHM2NetNode, validation_matrix: numpy.ndarray)
Represents the nets constructed by :class:`~.EHM2`.

Parameters
----------
root: :class:`~.EHM2NetNode`
    The net root node.
validation_matrix: :class:`numpy.ndarray`
    An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
    (aka. valid) associations between tracks and detections. The first column corresponds
    to the null hypothesis (hence contains all ones).
)";
    }

    std::string EHM2Net_num_layers()
    {
        return R"(Number of layers in the net)";
    }

    std::string EHM2Net_num_nodes()
    {
        return R"(Number of nodes in the net)";
    }

    std::string EHM2Net_root()
    {
        return R"(The root node of the net)";
    }

    std::string EHM2Net_nodes()
    {
        return R"(The nodes comprising the net)";
    }

    std::string EHM2Net_nodes_forward()
    {
        return R"(The net nodes, ordered by increasing layer)";
    }

    std::string EHM2Net_nodes_per_track()
    {
        return R"(Dictionary containing the nodes per track)";
    }

    std::string EHM2Net_add_node()
    {
        return R"(add_node(node: EHM2NetNode, parent: EHM2NetNode, detection: int)
Add a new node in the network.

Parameters
----------
node: :class:`~.EHM2NetNode`
    The node to be added.
parent: :class:`~.EHM2NetNode`
    The parent of the node.
detection: :class:`int`
    Index of measurement representing the parent child relationship.
)";
    }

    std::string EHM2Net_add_edge()
    {
        return R"(add_edge(parent: EHM2NetNode, child: EHM2NetNode, detection: int)
Add edge between two nodes, or update an already existing edge by adding the detection to it.

Parameters
----------
parent: :class:`~.EHM2NetNode`
    The parent node, i.e. the source of the edge.
child: :class:`~.EHM2NetNode`
    The child node, i.e. the target of the edge.
detection: :class:`int`
    Index of measurement representing the parent child relationship.
)";
    }

    std::string EHM2Net_get_nodes_per_layer_subnet()
    {
        return R"(get_nodes_per_layer_subnet(layer: int, subnet: int) -> List[EHM2NetNode]
Get nodes for a particular layer in a subnet.

Parameters
----------
layer: :class:`~.int`
    The target layer.
subnet: :class:`~.int`
    The target subnet.

Returns
-------
:class:`list` of :class:`~.EHM2NetNode`
    List of nodes in the target layer and subnet.
)";
    }

    std::string EHM2Net_get_children_per_detection()
    {
        return R"(get_children_per_detection(node: EHM2NetNode, detection: int) -> List[EHM2NetNode]
Get the children of a node for a particular detection.

Parameters
----------
node: :class:`~.EHM2NetNode`
    The node whose children should be returned.
detection: :class:`~.int`
    The target detection.
)";
    }

    std::string EHM2Tree()
    {
        return R"(EHM2Tree(track: int, children: List[EHM2Tree], detections: Set[int], subtree: int)
Represents the track tree structure generated by :func:`~pyehm.core.EHM2.construct_tree`.

The :class:`~.EHM2Tree` object represents both a tree as well as the root node in the tree.

Parameters
----------
track: :class:`int`
    The index of the track represented by the root node of the tree
children: :class:`list` of :class:`~.EHM2Tree`
    Sub-trees that are children of the current tree
detections: :class:`set` of :class:`int`
    Set of accumulated detections
subtree: :class:`int`
    Index of subtree the current tree belongs to.
)";
    }

    std::string EHM2Tree_depth()
    {
        return R"(The depth of the tree)";
    }

    std::string Cluster()
    {
        return R"(Cluster(tracks: List[int], detections: List[int] = [], validation_matrix: numpy.ndarray = numpy.array([]), likelihood_matrix: numpy.ndarray = numpy.array([]))
A cluster of tracks sharing common detections.

Parameters
----------
tracks: :class:`list` of `int`
    Indices of tracks in cluster
detections: :class:`list` of `int`
    Indices of detections in cluster. Defaults to an empty list.
validation_matrix: :class:`numpy.ndarray`
    The validation matrix for tracks and detections in the cluster. Defaults to an empty array.
likelihood_matrix: :class:`numpy.ndarray`
    The likelihood matrix for tracks and detections in the cluster. Defaults to an empty array.
)";
    }

    std::string gen_clusters()
    {
        return R"(gen_clusters(validation_matrix: numpy.ndarray, likelihood_matrix: numpy.ndarray = numpy.array([])) -> List[Cluster]
Cluster tracks into groups sharing detections

Parameters
----------
validation_matrix: :class:`numpy.ndarray`
    An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
    (aka. valid) associations between tracks and detections. The first column corresponds
    to the null hypothesis (hence contains all ones).
likelihood_matrix: :class:`numpy.ndarray`
    A matrix of shape (num_tracks, num_detections + 1) containing the unnormalised
    likelihoods for all combinations of tracks and detections. The first column corresponds
    to the null hypothesis. Defaults to an empty array, in which case the likelihood matrices of
    the generated clusters will also be empty arrays.
)";
    }

    std::string EHM()
    {
        return R"(Efficient Hypothesis Management (EHM)

An implementation of the EHM algorithm, as documented in [EHM1]_.
)";
    }

    std::string EHM_construct_net()
    {
        return R"(construct_net(validation_matrix: numpy.ndarray) -> EHMNet
Construct the EHM net as per Section 3.1 of [EHM1]_

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
)";
    }

    std::string EHM_compute_association_probabilities()
    {
        return R"(compute_association_probabilities(net: EHMNet, likelihood_matrix: numpy.ndarray) -> numpy.ndarray
Compute the joint association weights, as described in Section 3.3 of [EHM1]_

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
)";
    }

    std::string EHM_run()
    {
        return R"(run(validation_matrix: numpy.ndarray, likelihood_matrix: numpy.ndarray) -> numpy.ndarray
Run EHM to compute and return association probabilities

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
)";
    }

    std::string EHM2()
    {
        return R"(Efficient Hypothesis Management 2 (EHM2)

An implementation of the EHM2 algorithm, as documented in [EHM2]_.
)";
    }

    std::string EHM2_construct_net()
    {
        return R"(construct_net(validation_matrix: numpy.ndarray) -> EHM2Net
Construct the EHM2 net as per Section 4 of [EHM2]_

Parameters
----------
validation_matrix: :class:`numpy.ndarray`
    An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
    (aka. valid) associations between tracks and detections. The first column corresponds
    to the null hypothesis (hence contains all ones).
Returns
-------
: :class:`~.EHM2Net`
    The constructed net object
)";
    }

    std::string EHM2_construct_tree()
    {
        return R"(construct_tree(validation_matrix: numpy.ndarray) -> EHM2Tree
Construct the EHM2 tree as per section 4.3 of [EHM2]_

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
)";
    }

    std::string EHM2_compute_association_probabilities()
    {
        return R"(compute_association_probabilities(net: EHM2Net, likelihood_matrix: numpy.ndarray) -> numpy.ndarray
Compute the joint association weights, as described in Section 4.2 of [EHM2]_

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
)";
    }

    std::string EHM2_run()
    {
        return R"(run(validation_matrix: numpy.ndarray, likelihood_matrix: numpy.ndarray) -> numpy.ndarray
Run EHM2 to compute and return association probabilities

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
)";
    }


}