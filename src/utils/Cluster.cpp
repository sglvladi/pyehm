#include "Cluster.h"

namespace ehm
{
namespace utils
{

Cluster::Cluster(std::vector<int> tracks)
	:tracks(tracks)
{
}

Cluster::Cluster(std::vector<int> tracks, std::vector<int> detections, Eigen::MatrixXi validation_matrix)
	:tracks(tracks), detections(detections), validation_matrix(validation_matrix)
{
}

Cluster::Cluster(std::vector<int> tracks, std::vector<int> detections, Eigen::MatrixXi validation_matrix, Eigen::MatrixXd likelihood_matrix)
	:tracks(tracks), detections(detections), validation_matrix(validation_matrix), likelihood_matrix(likelihood_matrix)
{
}

} // namespace utils
} // namespace ehm