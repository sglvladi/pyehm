#include "EHMNet.h"

namespace ehm
{
namespace net
{

EHMNet::EHMNet(const EHMNetNodePtr root, const Eigen::MatrixXi& validation_matrix, const EHM2TreePtr tree)
{
    this->validation_matrix = validation_matrix;
    _nodes.push_back(root);
	_tree = tree;

	// Extract data from the tree
	auto tree_nodes = _tree->getNodes();
    for (auto& node : tree_nodes)
    {
        // Set up children layers per layer
		_child_layers[node->track] = {};
        for (auto& child : node->children)
        {
			_child_layers[node->track].push_back(child->track);
        }
		// Set up accumulated detections per layer
		_acc_per_layer[node->track] = node->detections;
	}

	_node_per_layer_identity[root->layer][root->identity].push_back(root->id);

}

void EHMNet::addNode(EHMNetNodePtr node, const EHMNetNodePtr parent, const int detection)
{
    // Set the node index
    node->id = _nodes.size();
    // Add the node to the network
    _nodes.push_back(node);
	// Add node to nodes per layer per identity
	_node_per_layer_identity[node->layer][node->identity].push_back(node->id);
}

void EHMNet::addEdge(const EHMNetNodePtr& parent, const EHMNetNodePtr& child, const int detection)
{    
    // TODO: Add later when optimise memory will be done
}

EHMNetNodePtr EHMNet::getRoot()
{
    return _nodes[0];
}

int EHMNet::getNumNodes()
{
    return _nodes.size();
}

int EHMNet::getNumLayers()
{
    return _num_layers;
}

std::vector<EHMNetNodePtr> EHMNet::getNodes()
{
    return _nodes;
}

std::vector<EHMNetNodePtr> EHMNet::getNodesForward()
{
    std::vector<EHMNetNodePtr> nodes_forward(_nodes.size());
    std::partial_sort_copy(_nodes.begin(), _nodes.end(), nodes_forward.begin(), nodes_forward.end(),
        [](const EHMNetNodePtr a, const EHMNetNodePtr b) {
			if (a->layer == -1 && b->layer != -1)
				return false;
			if (b->layer == -1 && a->layer != -1)
				return true;
			if (a->layer == b->layer)
				return a->id < b->id;
            return a->layer < b->layer;
        });
    return nodes_forward;
}

std::vector<EHMNetNodePtr> EHMNet::getNodesByLayer(int layer)
{
	std::vector<EHMNetNodePtr> nodes;
	for (const auto& pair : _node_per_layer_identity[layer])
	{
		const auto& node_ids = pair.second;
		for (const auto& node_id : node_ids)
		{
			nodes.push_back(_nodes[node_id]);
		}
	}
	return nodes;
}

std::vector<int> EHMNet::getChildLayers(int layer)
{
	return _child_layers[layer];
}

std::vector<EHMNetNodePtr> EHMNet::getChildrenPerDetection(const EHMNetNodePtr parent, const int detection)
{
	std::vector<EHMNetNodePtr> children;
	std::vector<int> child_layers = _child_layers[parent->layer];
	if (child_layers.size() == 0)
	{
		return getNodesByLayer(-1);
	}
	for (auto& child_layer : child_layers)
	{
		EHMNetNodeIdentity identity = computeIdentity(_acc_per_layer[child_layer], parent->identity, detection);
		for (auto& c_i : _node_per_layer_identity[child_layer][identity])
		{
			children.push_back(_nodes[c_i]);
		}
	}
	return children;
}




} // namespace utils
} // namespace ehm

