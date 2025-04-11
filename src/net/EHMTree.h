#pragma once
#include <set>
#include <vector>
#include <iostream>
#include <memory>

#include "EHMNetNode.h"

namespace ehm
{
namespace net
{

class EHMTree: public std::enable_shared_from_this<EHMTree>
{
    typedef std::shared_ptr<EHMTree> EHMTreePtr;
public:
    EHMTree() = default;
    EHMTree(int track, std::vector<EHMTreePtr> children, EHMNetNodeIdentity detections);

    int track;
    std::vector<EHMTreePtr> children;
    EHMNetNodeIdentity detections;

    int getDepth() const;
    std::vector<EHMTreePtr> getNodes();
    std::vector<EHMTreePtr> getAllChildren();
};

typedef std::shared_ptr<EHMTree> EHMTreePtr;

} // namespace utils
} // namespace ehm