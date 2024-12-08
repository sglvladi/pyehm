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

} // namespace utils
} // namespace ehm