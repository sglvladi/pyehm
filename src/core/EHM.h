#pragma once
#include<numeric>
#include <stack>

#include "../net/EHMNet.h"
#include "../utils/Utils.h"
#include "../utils/Cluster.h"


namespace ehm 
{
namespace core 
{

using namespace ehm::utils;
using namespace ehm::net;

class EHM
{
public:
    EHM();
    EHMNetPtr constructNet(const Eigen::MatrixXd& log_likelihood_matrix);
    Eigen::MatrixXd computeAssociationMatrix(const EHMNetPtr net, const Eigen::MatrixXd& log_likelihood_matrix);
    Eigen::MatrixXd run(const Eigen::MatrixXd& log_likelihood_matrix);
    virtual EHMTreePtr constructTree(const Eigen::MatrixXd& log_likelihood_matrix);
};

} // namespace core
} // namespace ehm


