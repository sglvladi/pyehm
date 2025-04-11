#include "EHM2.h"


namespace ehm
{
namespace core
{

EHM2::EHM2()
{
}

EHM2TreePtr EHM2::constructTree(const Eigen::MatrixXi& validation_matrix)
{
    int num_tracks = validation_matrix.rows();
    int num_detections = validation_matrix.cols();

    std::vector<EHM2TreePtr> trees;

    for (int i = num_tracks - 1; i >= 0; i--) {
        // Get indices of hypothesised detections for the track (minus the null hypothesis)
        std::set<int> v_detections;
        for (int detection = 1; detection < num_detections; detection++)
        {
            if (validation_matrix(i, detection) == 1)
            {
                v_detections.insert(detection);
            }
        }

        std::vector<int> matched;
        std::vector<int> unmatched;
        for (int j = 0; j < trees.size(); j++) {
            EHM2TreePtr tree = trees[j];
            std::set<int> tree_detections = tree->detections;
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

        std::vector<EHM2TreePtr> children;
        if (!matched.empty()) {
            for (int j = 0; j < matched.size(); j++) {
                children.push_back(trees[matched[j]]);
            }
            std::set<int> detections(v_detections.begin(), v_detections.end());
            for (auto& tree : children) {
                std::set_union(detections.begin(), detections.end(), tree->detections.begin(), tree->detections.end(), std::inserter(detections, detections.begin()));
            }
            EHM2TreePtr tree = std::make_shared<EHM2Tree>(EHM2Tree(i, children, detections));

            std::vector<EHM2TreePtr> new_trees;
            for (auto& j : unmatched) {
                new_trees.push_back(trees[j]);
            }
            new_trees.push_back(tree);
            trees = new_trees;
        }
        else {
            EHM2TreePtr tree = std::make_shared<EHM2Tree>(EHM2Tree(i, children, v_detections));
            trees.push_back(tree);
        }
    }

    // TODO: Add error if trees.size() != 1

    EHM2TreePtr tree = trees[0];

    return tree;
}

} // namespace core
} // namespace ehm
