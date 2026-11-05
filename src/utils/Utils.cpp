#include "Utils.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace ehm
{
namespace utils
{

TrackHypotheses toTrackHypotheses(const std::vector<TrackHypothesisArrays>& track_hypotheses)
{
	TrackHypotheses hypotheses;
	for (const auto& [detections, log_likelihoods] : track_hypotheses)
	{
		if (detections.size() != log_likelihoods.size()) {
			throw std::invalid_argument("Each track must have one log-likelihood per detection id");
		}
		std::vector<int> sorted_detections(detections.begin(), detections.end());
		std::sort(sorted_detections.begin(), sorted_detections.end());
		if (!sorted_detections.empty() && sorted_detections.front() < 0) {
			throw std::invalid_argument("Detection ids must not be negative");
		}
		if (std::adjacent_find(sorted_detections.begin(), sorted_detections.end()) != sorted_detections.end()) {
			throw std::invalid_argument("Each detection id must appear at most once per track");
		}
		hypotheses.detections.insert(hypotheses.detections.end(), detections.begin(), detections.end());
		hypotheses.log_likelihoods.insert(hypotheses.log_likelihoods.end(), log_likelihoods.begin(), log_likelihoods.end());
		hypotheses.offsets.push_back(hypotheses.detections.size());
	}
	return hypotheses;
}

TrackHypotheses toTrackHypotheses(const Eigen::MatrixXd& log_likelihood_matrix)
{
	TrackHypotheses hypotheses;
	for (int i = 0; i < log_likelihood_matrix.rows(); i++)
	{
		for (int j = 0; j < log_likelihood_matrix.cols(); j++)
		{
			// The null hypothesis is always kept, so that it can be assigned to tracks with no valid detections
			if (j == 0 || log_likelihood_matrix(i, j) != LOG_ZERO)
			{
				hypotheses.detections.push_back(j);
				hypotheses.log_likelihoods.push_back(log_likelihood_matrix(i, j));
			}
		}
		hypotheses.offsets.push_back(hypotheses.detections.size());
	}
	return hypotheses;
}

static int countIntersection(const std::vector<int>& sorted_a, const std::vector<int>& sorted_b)
{
	int count = 0;
	auto a = sorted_a.begin();
	auto b = sorted_b.begin();
	while (a != sorted_a.end() && b != sorted_b.end()) {
		if (*a < *b) {
			++a;
		}
		else if (*b < *a) {
			++b;
		}
		else {
			count++;
			++a;
			++b;
		}
	}
	return count;
}

// Order the tracks of a cluster by repeatedly merging the two groups of tracks sharing the most detections. This
// keeps tracks with common detections close together, which can greatly reduce the size of the resulting net.
static std::vector<int> orderClusterTracks(const std::vector<int>& tracks, const TrackHypotheses& hypotheses)
{
	if (tracks.size() < 3) {
		return tracks;
	}

	std::vector<std::vector<int>> groups;
	std::vector<std::vector<int>> group_detections;
	for (int track : tracks)
	{
		std::vector<int> detections;
		for (Eigen::Index k = hypotheses.offsets[track]; k < hypotheses.offsets[track + 1]; k++)
		{
			if (hypotheses.detections[k] != 0 && hypotheses.log_likelihoods[k] != LOG_ZERO)
			{
				detections.push_back(hypotheses.detections[k]);
			}
		}
		std::sort(detections.begin(), detections.end());
		groups.push_back({ track });
		group_detections.push_back(detections);
	}

	// Get table of number of intersections between groups
	int num_groups = groups.size();
	Eigen::MatrixXi num_intersects_table = Eigen::MatrixXi::Zero(num_groups, num_groups);
	for (int i1 = 0; i1 < num_groups - 1; i1++)
	{
		for (int i2 = i1 + 1; i2 < num_groups; i2++)
		{
			num_intersects_table(i1, i2) = countIntersection(group_detections[i1], group_detections[i2]);
		}
	}

	// All tracks of a cluster are connected, so there is always a pair of groups to merge
	while (num_groups > 1) {
		Eigen::Index maxi, maxj; // Use Eigen::Index for indexing
		num_intersects_table.maxCoeff(&maxi, &maxj);

		// Merge one group into another and delete it
		groups[maxi].insert(groups[maxi].end(), groups[maxj].begin(), groups[maxj].end());
		std::vector<int> merged_detections;
		std::set_union(group_detections[maxi].begin(), group_detections[maxi].end(), group_detections[maxj].begin(), group_detections[maxj].end(), std::back_inserter(merged_detections));
		group_detections[maxi] = std::move(merged_detections);
		groups.erase(groups.begin() + maxj);
		group_detections.erase(group_detections.begin() + maxj);

		// Remove the deleted group from the table and update the entries of the merged group
		num_groups--;
		Eigen::Index num_moved = num_groups - maxj;
		num_intersects_table.block(maxj, 0, num_moved, num_groups + 1) = num_intersects_table.block(maxj + 1, 0, num_moved, num_groups + 1).eval();
		num_intersects_table.block(0, maxj, num_groups + 1, num_moved) = num_intersects_table.block(0, maxj + 1, num_groups + 1, num_moved).eval();
		num_intersects_table.conservativeResize(num_groups, num_groups);
		for (int i = 0; i < num_groups; i++)
		{
			if (i != maxi)
			{
				int count = countIntersection(group_detections[i], group_detections[maxi]);
				num_intersects_table(std::min<Eigen::Index>(i, maxi), std::max<Eigen::Index>(i, maxi)) = count;
			}
		}
	}

	return groups[0];
}

std::vector<ClusterPtr> genClusters(const TrackHypotheses& hypotheses)
{
	int num_tracks = hypotheses.numTracks();

	// Disjoint sets of tracks sharing measurements
	std::vector<int> parent(num_tracks);
	std::iota(parent.begin(), parent.end(), 0);
	auto find_root = [&parent](int track) {
		while (parent[track] != track) {
			parent[track] = parent[parent[track]];
			track = parent[track];
		}
		return track;
	};

	// Merge the sets of tracks that hypothesise the same detection
	std::vector<bool> has_detections(num_tracks, false);
	std::unordered_map<int, int> track_per_detection;
	for (int track = 0; track < num_tracks; track++)
	{
		for (Eigen::Index k = hypotheses.offsets[track]; k < hypotheses.offsets[track + 1]; k++)
		{
			if (hypotheses.detections[k] == 0 || hypotheses.log_likelihoods[k] == LOG_ZERO)
			{
				continue;
			}
			has_detections[track] = true;
			auto [it, inserted] = track_per_detection.try_emplace(hypotheses.detections[k], track);
			if (!inserted)
			{
				parent[find_root(track)] = find_root(it->second);
			}
		}
	}

	// Group tracks per set, with clusters ordered by their first track
	std::vector<int> missed_tracks;
	std::vector<int> cluster_per_root(num_tracks, -1);
	std::vector<std::vector<int>> clusters;
	for (int track = 0; track < num_tracks; track++)
	{
		if (!has_detections[track])
		{
			missed_tracks.push_back(track);
			continue;
		}
		int root = find_root(track);
		if (cluster_per_root[root] == -1)
		{
			cluster_per_root[root] = clusters.size();
			clusters.emplace_back();
		}
		clusters[cluster_per_root[root]].push_back(track);
	}

	std::vector<ClusterPtr> clusters_obj;
	for (const auto& cluster_tracks : clusters)
	{
		std::vector<int> tracks = orderClusterTracks(cluster_tracks, hypotheses);

		// Get the sorted detections of the cluster, including the null hypothesis
		std::vector<int> v_detections_vec = { 0 };
		for (int track : tracks)
		{
			for (Eigen::Index k = hypotheses.offsets[track]; k < hypotheses.offsets[track + 1]; k++)
			{
				if (hypotheses.detections[k] != 0 && hypotheses.log_likelihoods[k] != LOG_ZERO)
				{
					v_detections_vec.push_back(hypotheses.detections[k]);
				}
			}
		}
		std::sort(v_detections_vec.begin(), v_detections_vec.end());
		v_detections_vec.erase(std::unique(v_detections_vec.begin(), v_detections_vec.end()), v_detections_vec.end());

		// Clusters are small relative to the full problem, so their log-likelihoods are stored as dense matrices
		Eigen::MatrixXd c_log_likelihood_matrix = Eigen::MatrixXd::Constant(tracks.size(), v_detections_vec.size(), LOG_ZERO);
		for (int i = 0; i < tracks.size(); i++)
		{
			for (Eigen::Index k = hypotheses.offsets[tracks[i]]; k < hypotheses.offsets[tracks[i] + 1]; k++)
			{
				if (hypotheses.log_likelihoods[k] != LOG_ZERO)
				{
					auto column = std::lower_bound(v_detections_vec.begin(), v_detections_vec.end(), hypotheses.detections[k]);
					c_log_likelihood_matrix(i, column - v_detections_vec.begin()) = hypotheses.log_likelihoods[k];
				}
			}
		}
		ClusterPtr cluster = std::make_shared<Cluster>(Cluster(tracks, v_detections_vec, c_log_likelihood_matrix));
		clusters_obj.push_back(cluster);
	}

	if (!missed_tracks.empty()) {
		ClusterPtr c = std::make_shared<Cluster>(Cluster(missed_tracks));
		clusters_obj.push_back(c);
	}

	return clusters_obj;
}

std::vector<ClusterPtr> genClusters(const Eigen::MatrixXd& log_likelihood_matrix)
{
	return genClusters(toTrackHypotheses(log_likelihood_matrix));
}

std::vector<int> computeIdentity(const std::set<int>& acc, const std::vector<int>& parent_identity, int detection)

{
	std::vector<int> identity;
	std::unordered_set<int> inter(parent_identity.begin(), parent_identity.end());
	inter.insert(detection);

	for (const int& elem : acc) {
		if (inter.find(elem) != inter.end()) {
			identity.push_back(elem);
		}
	}

	return identity;
}

std::unordered_set<int> computeRemainingDetections(const std::set<int>& v_detections, const std::vector<int>& parent_identity)
{
	std::unordered_set<int> v_detections_m1;
	std::set_difference(v_detections.begin(), v_detections.end(),
		parent_identity.begin(), parent_identity.end(),
		std::inserter(v_detections_m1, v_detections_m1.begin()));
	return v_detections_m1;
}

} // namespace utils
} // namespace ehm
