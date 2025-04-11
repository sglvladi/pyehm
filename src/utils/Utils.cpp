#include "Utils.h"

namespace ehm
{
namespace utils
{

Eigen::MatrixXi getNumIntersectsTable(const std::vector<std::pair<std::vector<int>, std::set<int>>>& clusters)
{
	int num_clusters = clusters.size();
	Eigen::MatrixXi num_intersects_table = Eigen::MatrixXi::Zero(num_clusters, num_clusters);
	for (int i1 = 0; i1 < num_clusters - 1; i1++)
	{
		for (int i2 = i1 + 1; i2 < num_clusters; i2++)
		{
			std::vector<int> intersection;
			std::set_intersection(clusters[i1].second.begin(), clusters[i1].second.end(), clusters[i2].second.begin(), clusters[i2].second.end(), std::back_inserter(intersection));
			num_intersects_table(i1, i2) = intersection.size();
		}
	}
	return num_intersects_table;
}

std::vector<ClusterPtr> genClusters(const Eigen::MatrixXi& validation_matrix, const Eigen::MatrixXd& likelihood_matrix)
{
    // Validation matrix for all detections except null
    auto validation_matrix_true = validation_matrix(Eigen::all, Eigen::seqN(1, Eigen::last));

    // Initiate parameters
    int num_tracks = validation_matrix_true.rows();
    int num_detections = validation_matrix_true.cols();

    // Form clusters of tracks sharing measurements
    std::set<int> missed_tracks;
    std::vector<std::pair<std::vector<int>, std::set<int>>> clusters;
    std::vector<ClusterPtr> clusters_obj;

	for (int i = 0; i < num_tracks; i++)
	{
		std::pair<std::vector<int>, std::set<int>> cluster;
		cluster.first.push_back(i);
		for (int j = 0; j < num_detections; j++)
		{
			if (validation_matrix_true(i, j) == 1)
			{
				cluster.second.insert(j+1);
			}
		}
		clusters.push_back(cluster);
	}
    
	// Get table of number of intersections between clusters
	Eigen::MatrixXi num_intersects_table = getNumIntersectsTable(clusters);

    // Continue until we have only one cluster or none of them intersect
    while (!clusters.empty()) {
        // Find maximum intersection - if no intersection, break
        Eigen::Index maxi, maxj; // Use Eigen::Index for indexing
        int maxVal = num_intersects_table.maxCoeff(&maxi, &maxj);
		if (maxVal == 0) {
			break;
		}

        // Merge one cluster into another and delete it
		std::vector<int> merged_tracks;
		std::set<int> merged_detections;
		merged_tracks.insert(merged_tracks.end(), clusters[maxi].first.begin(), clusters[maxi].first.end());
		merged_tracks.insert(merged_tracks.end(), clusters[maxj].first.begin(), clusters[maxj].first.end());
		std::set_union(clusters[maxi].second.begin(), clusters[maxi].second.end(), clusters[maxj].second.begin(), clusters[maxj].second.end(), std::inserter(merged_detections, merged_detections.begin()));
		clusters[maxi] = std::make_pair(merged_tracks, merged_detections);
		clusters.erase(clusters.begin() + maxj);

		// Update table of number of intersections between clusters
		num_intersects_table = getNumIntersectsTable(clusters);
    }

	for (auto& clust : clusters) {
		if (clust.second.size() == 0) {
			missed_tracks.insert(clust.first[0]);
			continue;
		}
		clust.second.insert(0);
		std::vector<int> v_detections_vec(clust.second.begin(), clust.second.end());
		Eigen::MatrixXi c_validation_matrix = validation_matrix(clust.first, v_detections_vec);
		Eigen::MatrixXd c_likelihood_matrix;
		if (likelihood_matrix.rows() == 0) {
			c_likelihood_matrix = Eigen::MatrixXd::Zero(0, 0);
		}
		else {
			c_likelihood_matrix = likelihood_matrix(clust.first, v_detections_vec);
		}
		ClusterPtr cluster = std::make_shared<Cluster>(Cluster(clust.first, v_detections_vec, c_validation_matrix, c_likelihood_matrix));
		clusters_obj.push_back(cluster);
	}

	if (!missed_tracks.empty()) {
		std::vector<int> missed_tracks_vec(missed_tracks.begin(), missed_tracks.end());
		ClusterPtr c = std::make_shared<Cluster>(Cluster(missed_tracks_vec));
		clusters_obj.push_back(c);
	}

    return clusters_obj;
}

std::set<int> computeIdentity(const std::set<int>& acc, const std::set<int>& parent_identity, int detection)

{
	std::set<int> identity;
	std::unordered_set<int> inter(parent_identity.begin(), parent_identity.end());
	inter.insert(detection);

	for (const int& elem : acc) {
		if (inter.find(elem) != inter.end()) {
			identity.insert(elem);
		}
	}

	return identity;
}

std::unordered_set<int> computeRemainingDetections(const std::set<int>& v_detections, const std::set<int>& parent_identity)
{
	std::unordered_set<int> v_detections_m1;
	std::set_difference(v_detections.begin(), v_detections.end(),
		parent_identity.begin(), parent_identity.end(),
		std::inserter(v_detections_m1, v_detections_m1.begin()));
	return v_detections_m1;
}

} // namespace utils
} // namespace ehm
