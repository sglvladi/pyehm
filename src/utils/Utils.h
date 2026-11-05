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

// Detection ids and log-likelihoods of the hypotheses of a single track
typedef std::pair<Eigen::VectorXi, Eigen::VectorXd> TrackHypothesisArrays;

// Hypotheses of all tracks, stored in flat arrays. The hypotheses of track i are the entries in the
// range [offsets[i], offsets[i + 1]) of detections and log_likelihoods. Detection id 0 is the null hypothesis.
struct TrackHypotheses
{
    std::vector<Eigen::Index> offsets = { 0 };
    std::vector<int> detections;
    std::vector<double> log_likelihoods;

    int numTracks() const { return offsets.size() - 1; }
};

TrackHypotheses toTrackHypotheses(const std::vector<TrackHypothesisArrays>& track_hypotheses);
TrackHypotheses toTrackHypotheses(const Eigen::MatrixXd& log_likelihood_matrix);
std::vector<ClusterPtr> genClusters(const TrackHypotheses& hypotheses);
std::vector<ClusterPtr> genClusters(const Eigen::MatrixXd& log_likelihood_matrix);
std::vector<int> computeIdentity(const std::set<int>& acc, const std::vector<int>& parent_identity, int detection);
std::unordered_set<int> computeRemainingDetections(const std::set<int>& v_detections, const std::vector<int>& parent_identity);

} // namespace utils
} // namespace ehm