#include "EHMTree.h"

namespace ehm
{
namespace net
{

EHMTree::EHMTree(int track, std::vector<EHMTreePtr> children, EHMNetNodeIdentity detections)
    :track(track), children(children), detections(detections)
{
}

int EHMTree::getDepth() const
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

std::vector<EHMTreePtr> EHMTree::getNodes()
{
	std::vector<EHMTreePtr> nodes;
	nodes.push_back(std::make_shared<EHMTree>(*this));
	for (auto& child : children)
	{
		std::vector<EHMTreePtr> c_nodes = child->getNodes();
		nodes.insert(nodes.end(), c_nodes.begin(), c_nodes.end());
	}
    return nodes;
}

std::vector<EHMTreePtr> EHMTree::getAllChildren()
{
    std::vector<EHMTreePtr> nodes;
    for (auto& child : children)
    {    
        nodes.push_back(child);
        std::vector<EHMTreePtr> c_nodes = child->getAllChildren();
        nodes.insert(nodes.end(), c_nodes.begin(), c_nodes.end());
    }
    return nodes;
}

} // namespace utils
} // namespace ehm
