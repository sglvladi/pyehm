#pragma once
#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace ehm
{
namespace utils
{

class Cluster
{
public:
    Cluster(std::vector<int> tracks);
    Cluster(std::vector<int> tracks, std::vector<int> detections, Eigen::MatrixXi validation_matrix);
    Cluster(std::vector<int> tracks, std::vector<int> detections, Eigen::MatrixXi validation_matrix, Eigen::MatrixXd likelihood_matrix);
    
    std::vector<int> tracks;
    std::vector<int> detections;
    Eigen::MatrixXi validation_matrix;
    Eigen::MatrixXd likelihood_matrix;
};

typedef std::shared_ptr<Cluster> ClusterPtr;

} // namespace utils
} // namespace ehm
