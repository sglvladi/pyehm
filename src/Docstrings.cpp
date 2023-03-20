#include "Docstrings.h"
namespace docstrings {

	std::string EHMNetNode() {
        return R"mydelimeter(
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
)mydelimeter";
	}

    std::string EHM2NetNode() {
        return R"mydelimeter(
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
)mydelimeter";
    }

    std::string EHMNet()
    {
        return R"(
Represents the nets constructed by :class:`~.EHM`.

Parameters
----------
root: ::class:`~.EHMNetNode`
    The net root node.
validation_matrix: :class:`numpy.ndarray`
    An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
    (aka. valid) associations between tracks and detections. The first column corresponds
    to the null hypothesis (hence contains all ones).
)";
    }

    std::string EHMNet_num_layers()
    {
        return R"(Number of layers in the net)";
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
        return R"(
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
        return R"(
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
        return R"(
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
        return R"(
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
        return R"(
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
        return R"(
Represents the nets constructed by :class:`~.EHM2`.

Parameters
----------
root: ::class:`~.EHM2NetNode`
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

    std::string EHM2Net_add_node()
    {
        return R"(
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
        return R"(
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
        return R"(
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


}