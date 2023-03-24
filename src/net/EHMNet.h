#pragma once
#include <map>
#include <unordered_map>
#include <vector>
#include <memory>

#include <Eigen/Dense>

#include "EHMNetNode.h"

namespace ehm
{
namespace net
{

template <typename T> 
class EHMNetBase {

public:
	Eigen::MatrixXi validation_matrix;

	const T getRoot() {
		return _nodes[0];
	}
	const int getNumNodes() {
		return _nodes.size();
	}
	const int getNumLayers() {
		return _num_layers;
	}
	const std::vector<T> getNodes() {
		return _nodes;
	}
	const std::vector<T> getNodesForward() {
		std::vector<T> nodes_forward(_nodes.size());
		std::partial_sort_copy(_nodes.begin(), _nodes.end(), nodes_forward.begin(), nodes_forward.end(), 
			[](const T a, const T b) {
				return a->layer < b->layer;
			});
		return nodes_forward;
	}


	virtual void addNode(T node, const T parent, const int detection) = 0;
	virtual void addEdge(const T& parent, const T& child, const int detection) = 0;

protected:
	int _num_layers = 0;
	std::vector<T> _nodes;

};

class EHMNet: public EHMNetBase<EHMNetNodePtr>
{
public:

	EHMNet() = default;
	EHMNet(const EHMNetNodePtr root, const Eigen::MatrixXi& validation_matrix);


	const std::set<EHMNetNodePtr> getParents(const EHMNetNodePtr node);
	const std::set<EHMNetNodePtr> getChildren(const EHMNetNodePtr node);
	const std::set<int> getEdges(const EHMNetNodePtr parent, const EHMNetNodePtr child);

	void addNode(EHMNetNodePtr node, const EHMNetNodePtr parent, const int detection);
	void addEdge(const EHMNetNodePtr& parent, const EHMNetNodePtr& child, const int detection);

private:
	std::map<std::pair<int, int>, std::set<int>> _edges;
	std::map<int, std::set<EHMNetNodePtr>> _parents;
	std::map<int, std::set<EHMNetNodePtr>> _children;
};

typedef std::shared_ptr<EHMNet> EHMNetPtr;

} // namespace utils
} // namespace ehm
