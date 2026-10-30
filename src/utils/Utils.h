#pragma once
#include <cmath>
#include <limits>
#include <set>
#include <unordered_set>
#include "Cluster.h"
#include <Eigen/Dense>

namespace ehm
{
namespace utils
{

// Log of zero, used to denote invalid associations in log-likelihood matrices
constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();

// Numerically stable log(exp(a) + exp(b))
inline double logAdd(double a, double b)
{
    if (a == LOG_ZERO) return b;
    if (b == LOG_ZERO) return a;
    return a > b ? a + std::log1p(std::exp(b - a)) : b + std::log1p(std::exp(a - b));
}

Eigen::MatrixXi getNumIntersectsTable(const std::vector<std::pair<std::vector<int>,std::set<int>>>& clusters);
std::vector<ClusterPtr> genClusters(const Eigen::MatrixXd& log_likelihood_matrix);
std::vector<int> computeIdentity(const std::set<int>& acc, const std::vector<int>& parent_identity, int detection);
std::unordered_set<int> computeRemainingDetections(const std::set<int>& v_detections, const std::vector<int>& parent_identity);

} // namespace utils
} // namespace ehm