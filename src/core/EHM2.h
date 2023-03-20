#pragma once
#include <numeric>
#include <stack>

#include "../utils/EHM2Tree.h"
#include "../utils/EHM2Net.h"
#include "../utils/Utils.h"
#include "../utils/Cluster.h"

namespace ehm
{
namespace core
{

using namespace ehm::utils;

class EHM2
{
public:
	EHM2();
	static EHM2NetPtr constructNet(const Eigen::MatrixXi& validation_matrix);
	static Eigen::MatrixXd computeAssociationMatrix(const EHM2NetPtr net, const Eigen::MatrixXd& likelihood_matrix);
	static Eigen::MatrixXd run(const Eigen::MatrixXi& validation_matrix, const Eigen::MatrixXd& likelihood_matrix);
	static EHM2TreePtr constructTree(const Eigen::MatrixXi& validation_matrix);
};

} // namespace core
} // namespace ehm
