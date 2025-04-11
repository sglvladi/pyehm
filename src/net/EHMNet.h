#pragma once
#include <map>
#include <unordered_map>
#include <vector>
#include <memory>

#include <Eigen/Dense>

#include "EHMNetNode.h"
#include "EHMTree.h"
#include "../utils/Utils.h"

namespace ehm
{
namespace net
{

using namespace ehm::utils;

class EHMNet
{
public:
    Eigen::MatrixXi validation_matrix;

    EHMNet() = default;
    EHMNet(const EHMNetNodePtr root, const Eigen::MatrixXi& validation_matrix, const EHMTreePtr tree);

    void addNode(EHMNetNodePtr node, const EHMNetNodePtr parent, const int detection);
    void addEdge(const EHMNetNodePtr& parent, const EHMNetNodePtr& child, const int detection);
    EHMNetNodePtr getRoot();
    int getNumNodes();
    int getNumLayers();
    std::vector<EHMNetNodePtr> getNodes();
    std::vector<EHMNetNodePtr> getNodesForward();
    std::vector<EHMNetNodePtr> getNodesByLayer(int layer);
	std::vector<int> getChildLayers(int layer);
	std::vector<EHMNetNodePtr> getChildrenPerDetection(const EHMNetNodePtr parent, const int detection);

private:
    int _num_layers = 0;
    std::vector<EHMNetNodePtr> _nodes;
	EHMTreePtr _tree;
    std::map<int, std::map<EHMNetNodeIdentity, std::vector<int>>> _node_per_layer_identity;
    std::map<int, std::vector<int>> _child_layers;
	std::map<int, std::set<int>> _acc_per_layer;

};

typedef std::shared_ptr<EHMNet> EHMNetPtr;

} // namespace utils
} // namespace ehm
