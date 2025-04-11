#pragma once
#include <numeric>
#include <stack>

#include "EHM.h"
#include "../net/EHMTree.h"
#include "../utils/Utils.h"
#include "../utils/Cluster.h"

namespace ehm
{
namespace core
{

using namespace ehm::utils;
using namespace ehm::net;

class EHM2: public EHM
{
public:
    EHM2();
    EHMTreePtr constructTree(const Eigen::MatrixXi& validation_matrix) override;
};

} // namespace core
} // namespace ehm
