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
    EHMNetPtr constructNet(const Eigen::MatrixXi& validation_matrix);
    Eigen::MatrixXd computeAssociationMatrix(const EHMNetPtr net, const Eigen::MatrixXd& likelihood_matrix);
    Eigen::MatrixXd run(const Eigen::MatrixXi& validation_matrix, const Eigen::MatrixXd& likelihood_matrix);
    virtual EHM2TreePtr constructTree(const Eigen::MatrixXi& validation_matrix);
};

} // namespace core
} // namespace ehm


