#pragma once
#include <set>
#include "Cluster.h"
#include <Eigen/Dense>

namespace ehm
{
namespace utils
{

void dfs(int vertex, const std::vector<std::vector<int>>& graph, std::vector<bool>& visited, std::vector<int>& component);
std::vector<std::vector<int>> findConnectedComponents(const std::vector<std::vector<int>>& graph);
std::vector<ClusterPtr> genClusters(const Eigen::MatrixXi& validation_matrix, const Eigen::MatrixXd& likelihood_matrix);

} // namespace utils
} // namespace ehm