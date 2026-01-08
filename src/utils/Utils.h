#pragma once
#include <set>
#include <unordered_set>
#include "Cluster.h"
#include <Eigen/Dense>

namespace ehm
{
namespace utils
{

Eigen::MatrixXi getNumIntersectsTable(const std::vector<std::pair<std::vector<int>,std::set<int>>>& clusters);
std::vector<ClusterPtr> genClusters(const Eigen::MatrixXi& validation_matrix, const Eigen::MatrixXd& likelihood_matrix);
std::vector<int> computeIdentity(const std::set<int>& acc, const std::vector<int>& parent_identity, int detection);
std::unordered_set<int> computeRemainingDetections(const std::set<int>& v_detections, const std::vector<int>& parent_identity);

} // namespace utils
} // namespace ehm