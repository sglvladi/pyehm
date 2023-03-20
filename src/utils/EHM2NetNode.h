#pragma once
#include "EHMNetNode.h"

namespace ehm
{
namespace utils
{

class EHM2NetNode : public EHMNetNode
{

public:
	int track;
	int subnet;

	EHM2NetNode() = default;
	EHM2NetNode(int layer, int track, int subnet, EHMNetNodeIdentity identity);
	EHM2NetNode(int layer, EHMNetNodeIdentity identity, int track, int subnet);
	EHM2NetNode(int layer);

	virtual std::string toString() const override;
};

typedef std::shared_ptr<EHM2NetNode> EHM2NetNodePtr;

} // namespace utils
} // namespace ehm