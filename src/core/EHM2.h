#pragma once
#include <numeric>
#include <stack>

#include "EHM.h"
#include "../net/EHM2Tree.h"
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
    static EHM2TreePtr constructTree(const Eigen::MatrixXi& validation_matrix);
};

} // namespace core
} // namespace ehm
