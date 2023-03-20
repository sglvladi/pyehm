#pragma once
#include <unordered_map>
#include "EHMNet.h"
#include "EHM2NetNode.h"
#include "Utils.h"

namespace ehm
{
namespace utils
{

class EHM2Net : public EHMNetBase<EHM2NetNodePtr>
{
public:
	EHM2Net() = default;
	EHM2Net(const EHM2NetNodePtr root, const Eigen::MatrixXi& validation_matrix);

	std::set<EHM2NetNodePtr> getNodesPerLayerSubnet(const int layer, const int subnet);
	const std::vector<EHM2NetNodePtr> getChildrenPerDetection(const EHM2NetNodePtr parent, const int detection);
	void addNode(EHM2NetNodePtr node, const EHM2NetNodePtr parent, const int detection);
	void addEdge(const EHM2NetNodePtr& parent, const EHM2NetNodePtr& child, const int detection);

	std::map<int, std::set<EHM2NetNodePtr>> nodes_per_track;

private:
	std::map<std::pair<int, int>, std::set<EHM2NetNodePtr>> _nodes_per_layer_subnet;
	std::map<std::pair<int, int>, std::set<int>> _children_per_detection;

};

typedef std::shared_ptr<EHM2Net> EHM2NetPtr;

} // namespace utils
} // namespace ehm


