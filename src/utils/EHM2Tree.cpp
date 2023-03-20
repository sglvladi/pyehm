#include "EHM2Tree.h"

namespace ehm
{
namespace utils
{

EHM2Tree::EHM2Tree(int track, std::vector<EHM2TreePtr> children, EHMNetNodeIdentity detections, int subtree)
	:track(track), children(children), detections(detections), subtree(subtree)
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
