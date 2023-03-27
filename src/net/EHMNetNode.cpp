#include "EHMNetNode.h"

namespace ehm
{
namespace net
{

EHMNetNode::EHMNetNode(int layer, EHMNetNodeIdentity identity): layer(layer), identity(identity)
{
    id = 0;
}

EHMNetNode::EHMNetNode(int layer): layer(layer)
{
    id = 0;
    identity = {};
}

std::string EHMNetNode::toString() const
{
    std::string str = "EHMNetNode(id=" + std::to_string(id) + ", ";
    str += "layer=" + std::to_string(layer) + ", ";
    str += "identity={";
    std::vector<int> identity_vec(identity.begin(), identity.end());
    for (int i = 0; i < identity_vec.size(); i++)
    {
        str += std::to_string(identity_vec[i]);
        if (i != identity_vec.size() - 1) str += ", ";
    }
    str += "})";
    return str;
}

std::ostream& operator<<(std::ostream& os, const EHMNetNode& n)
{
    os << n.toString();
    return os;
}

} // namespace utils
} // namespace ehm
