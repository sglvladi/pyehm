#include "Cluster.h"

namespace ehm
{
namespace utils
{

Cluster::Cluster(std::vector<int> tracks)
    :tracks(tracks)
{
}

Cluster::Cluster(std::vector<int> tracks, std::vector<int> detections, Eigen::MatrixXd log_likelihood_matrix)
    :tracks(tracks), detections(detections), log_likelihood_matrix(log_likelihood_matrix)
{
}

} // namespace utils
} // namespace ehm