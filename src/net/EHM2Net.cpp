#include "EHM2Net.h"

namespace ehm
{
namespace net
{

EHM2Net::EHM2Net(const EHM2NetNodePtr root, const Eigen::MatrixXi& validation_matrix)
{
	this->validation_matrix = validation_matrix;
	_nodes.push_back(root);
	_nodes_per_layer_subnet[std::make_pair(root->layer, root->subnet)].insert(root);
}

std::set<EHM2NetNodePtr> EHM2Net::getNodesPerLayerSubnet(const int layer, const int subnet)
{
	std::pair<int, int> key = std::make_pair(layer, subnet);
	if (_nodes_per_layer_subnet.find(key) == _nodes_per_layer_subnet.end())
	{
		return std::set<EHM2NetNodePtr>();
	}
	return _nodes_per_layer_subnet[std::make_pair(layer, subnet)];
}

const std::vector<EHM2NetNodePtr> EHM2Net::getChildrenPerDetection(const EHM2NetNodePtr parent, const int detection)
{
	std::pair<int, int> key = std::make_pair(parent->id, detection);
	if (_children_per_detection.find(key) == _children_per_detection.end())
	{
		return std::vector<EHM2NetNodePtr>();
	}
	std::set<int> children_ids = _children_per_detection[key];
	std::vector<EHM2NetNodePtr> children(children_ids.size());
	int i = 0;
	for (int id : children_ids)
	{
		children[i] = _nodes[id];
		i++;
	}
	return children;
}

void EHM2Net::addNode(EHM2NetNodePtr node, const EHM2NetNodePtr parent, const int detection)
{
	// Set the node index
	node->id = _nodes.size();
	// Add the node to the network
	_nodes.push_back(node);
	if (node->layer + 1 > _num_layers)
	{
		_num_layers = node->layer + 1;
	}
	_nodes_per_layer_subnet[std::make_pair(node->layer, node->subnet)].insert(node);
	_children_per_detection[std::make_pair(parent->id, detection)].insert(node->id);
}

void EHM2Net::addEdge(const EHM2NetNodePtr& parent, const EHM2NetNodePtr& child, const int detection)
{
	_children_per_detection[std::make_pair(parent->id, detection)].insert(child->id);
}

} // namespace utils
} // namespace ehm