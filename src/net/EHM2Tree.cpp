#include "EHM2Tree.h"

namespace ehm
{
namespace net
{

EHM2Tree::EHM2Tree(int track, std::vector<EHM2TreePtr> children, EHMNetNodeIdentity detections)
    :track(track), children(children), detections(detections)
{
}

int EHM2Tree::getDepth() const
{
    int depth = 1;
    int c_depth = 0;
    for (auto& child : children)
    {
        c_depth = std::max(c_depth, child->getDepth());
    }
    depth += c_depth;
    return depth;
}

std::vector<EHM2TreePtr> EHM2Tree::getNodes()
{
	std::vector<EHM2TreePtr> nodes;
	nodes.push_back(std::make_shared<EHM2Tree>(*this));
	for (auto& child : children)
	{
		std::vector<EHM2TreePtr> c_nodes = child->getNodes();
		nodes.insert(nodes.end(), c_nodes.begin(), c_nodes.end());
	}
    return nodes;
}

std::vector<EHM2TreePtr> EHM2Tree::getAllChildren()
{
    std::vector<EHM2TreePtr> nodes;
    for (auto& child : children)
    {    
        nodes.push_back(child);
        std::vector<EHM2TreePtr> c_nodes = child->getAllChildren();
        nodes.insert(nodes.end(), c_nodes.begin(), c_nodes.end());
    }
    return nodes;
}

} // namespace utils
} // namespace ehm
