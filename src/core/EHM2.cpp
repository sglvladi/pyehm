#include "EHM2.h"


namespace ehm
{
namespace core
{

EHM2::EHM2()
{
}

EHMTreePtr EHM2::constructTree(const Eigen::MatrixXi& validation_matrix)
{
    int num_tracks = validation_matrix.rows();
    int num_detections = validation_matrix.cols();

    std::vector<EHMTreePtr> trees;

    for (int i = num_tracks - 1; i >= 0; i--) {
        // Get indices of hypothesised detections for the track (minus the null hypothesis)
        std::vector<int> v_detections;
        for (int detection = 1; detection < num_detections; detection++)
        {
            if (validation_matrix(i, detection) == 1)
            {
                v_detections.push_back(detection);
            }
        }

        std::vector<int> matched;
        std::vector<int> unmatched;
        for (int j = 0; j < trees.size(); j++) {
            EHMTreePtr tree = trees[j];
            std::vector<int> tree_detections = tree->detections;
            std::set<int> intersection;
            std::set_intersection(v_detections.begin(), v_detections.end(), tree_detections.begin(), tree_detections.end(), std::inserter(intersection, intersection.begin()));
            if (!intersection.empty()) {
                // Add the track to the tree
                matched.push_back(j);
            }
            else {
                unmatched.push_back(j);
            }
        }

        std::vector<EHMTreePtr> children;
        if (!matched.empty()) {
            for (int j = 0; j < matched.size(); j++) {
                children.push_back(trees[matched[j]]);
            }
			// IMPORTANT: This needs to be a set to avoid out-of-order detections
            std::set<int> detections(v_detections.begin(), v_detections.end());
            for (auto& tree : children) {
                std::vector<int> tree_detections = tree->detections;
				std::set_union(detections.begin(), detections.end(), tree->detections.begin(), tree->detections.end(), std::inserter(detections, detections.begin()));
            }
			// Now that we have the (ordered!) union of all detections, we need to convert it back to a vector
			std::vector<int> detections_vec(detections.begin(), detections.end());
            EHMTreePtr tree = std::make_shared<EHMTree>(EHMTree(i, children, detections_vec));

            std::vector<EHMTreePtr> new_trees;
            for (auto& j : unmatched) {
                new_trees.push_back(trees[j]);
            }
            new_trees.push_back(tree);
            trees = new_trees;
        }
        else {
            EHMTreePtr tree = std::make_shared<EHMTree>(EHMTree(i, children, v_detections));
            trees.push_back(tree);
        }
    }

    // TODO: Add error if trees.size() != 1

    EHMTreePtr tree = trees[0];

    return tree;
}

} // namespace core
} // namespace ehm
