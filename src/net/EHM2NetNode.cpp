#include "EHM2NetNode.h"

namespace ehm
{
namespace net
{

EHM2NetNode::EHM2NetNode(int layer, int track, int subnet, EHMNetNodeIdentity identity)
	: EHMNetNode(layer, identity), track(track), subnet(subnet)
{
}

EHM2NetNode::EHM2NetNode(int layer, EHMNetNodeIdentity identity, int track, int subnet)
	: EHMNetNode(layer, identity), track(track), subnet(subnet)
{
}

EHM2NetNode::EHM2NetNode(int layer)
	: EHMNetNode(layer)
{
	track = -1;
	subnet = 0;
}

std::string EHM2NetNode::toString() const
{
	std::string str = "EHM2NetNode(id=" + std::to_string(id) + ", ";
	str += "layer=" + std::to_string(layer) + ", ";
	str += "identity={";
	std::vector<int> identity_vec(identity.begin(), identity.end());
	for (int i = 0; i < identity_vec.size(); i++)
	{
		str += std::to_string(identity_vec[i]);
		if (i != identity_vec.size() - 1) str += ", ";
	}
	str += "}, ";
	str += "track=" + std::to_string(track) + ", ";
	str += "subnet=" + std::to_string(subnet) + ")";
	return str;
}

} // namespace utils
} // namespace ehm