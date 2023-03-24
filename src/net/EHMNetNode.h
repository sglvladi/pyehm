#pragma once
#include <set>
#include <vector>
#include <string>
#include <iterator>
#include <iostream>
#include <memory>


namespace ehm
{
namespace net
{

typedef std::set<int> EHMNetNodeIdentity;

class EHMNetNode
{
public:
	int id;
	int layer;
	EHMNetNodeIdentity identity;

	EHMNetNode() = default;
	EHMNetNode(int layer, EHMNetNodeIdentity identity);
	EHMNetNode(int layer);

	virtual std::string toString() const;

	friend std::ostream& operator<<(std::ostream& os, const EHMNetNode& n);

};


typedef std::shared_ptr<EHMNetNode> EHMNetNodePtr;

} // namespace utils
} // namespace ehm