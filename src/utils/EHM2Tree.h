#pragma once
#include <set>
#include <vector>
#include <iostream>
#include <memory>

#include "EHMNetNode.h"

namespace ehm
{
namespace utils
{

class EHM2Tree: public std::enable_shared_from_this<EHM2Tree>
{
	typedef std::shared_ptr<EHM2Tree> EHM2TreePtr;
public:
	EHM2Tree() = default;
	EHM2Tree(int track, std::vector<EHM2TreePtr> children, EHMNetNodeIdentity detections, int subtree);

	int track;
	std::vector<EHM2TreePtr> children;
	EHMNetNodeIdentity detections;
	int subtree;

	int getDepth() const;
	std::vector<EHM2TreePtr> getAllChildren();
};

typedef std::shared_ptr<EHM2Tree> EHM2TreePtr;

} // namespace utils
} // namespace ehm