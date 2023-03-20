#include "EHMNet.h"

namespace ehm
{
namespace utils
{

EHMNet::EHMNet(const EHMNetNodePtr root, const Eigen::MatrixXi& validation_matrix)
{
	this->validation_matrix = validation_matrix;
	_nodes.push_back(root);
}

const std::set<EHMNetNodePtr> EHMNet::getParents(const EHMNetNodePtr node)
{
	if (_parents.find(node->id) == _parents.end()) {
		return std::set<EHMNetNodePtr>();
	}
	return _parents[node->id];
}

const std::set<EHMNetNodePtr> EHMNet::getChildren(const EHMNetNodePtr node)
{
	if (_children.find(node->id) == _children.end()) {
		return std::set<EHMNetNodePtr>();
	}
	return _children[node->id];
}

const std::set<int> EHMNet::getEdges(const EHMNetNodePtr parent, const EHMNetNodePtr child)
{
	std::pair<int, int> key = std::make_pair(parent->id, child->id);
	if (_edges.find(key) == _edges.end()) {
		return std::set<int>();
	}
	return _edges[key];
}

void EHMNet::addNode(EHMNetNodePtr node, const EHMNetNodePtr parent, const int detection)
{
	// Set the node index
	node->id = _nodes.size();
	// Add the node to the network
	_nodes.push_back(node);
	// Create edge from parent to child
	_edges[std::make_pair(parent->id, node->id)].insert(detection);
	// Add the child to the parent's children
	_parents[node->id].insert(parent);
	// Add the parent to the child's parents
	_children[parent->id].insert(node);
	if (node->layer + 2 > _num_layers)
	{
		_num_layers = node->layer + 2;
	}
	
}

void EHMNet::addEdge(const EHMNetNodePtr& parent, const EHMNetNodePtr& child, const int detection)
{	
	_edges[std::make_pair(parent->id, child->id)].insert(detection);
	_children[parent->id].insert(child);
	_parents[child->id].insert(parent);
}

} // namespace utils
} // namespace ehm

