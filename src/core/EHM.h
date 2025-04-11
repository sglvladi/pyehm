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
    static EHMNetPtr constructNet(const Eigen::MatrixXi& validation_matrix);
    static Eigen::MatrixXd computeAssociationMatrix(const EHMNetPtr net, const Eigen::MatrixXd& likelihood_matrix);
    static Eigen::MatrixXd run(const Eigen::MatrixXi& validation_matrix, const Eigen::MatrixXd& likelihood_matrix);
    static EHM2TreePtr constructTree(const Eigen::MatrixXi& validation_matrix);
};

} // namespace core
} // namespace ehm


