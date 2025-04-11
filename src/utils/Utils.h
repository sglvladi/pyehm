#pragma once
#include <set>
#include "Cluster.h"
#include <Eigen/Dense>

namespace ehm
{
namespace utils
{

Eigen::MatrixXi getNumIntersectsTable(std::vector<std::pair<std::vector<int>,std::set<int>>> clusters);
std::vector<ClusterPtr> genClusters(const Eigen::MatrixXi& validation_matrix, const Eigen::MatrixXd& likelihood_matrix);
std::set<int> computeIdentity(const std::set<int> acc, const std::set<int> parent_identity, int detection);

} // namespace utils
} // namespace ehm