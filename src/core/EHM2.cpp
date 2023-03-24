#include "EHM2.h"


namespace ehm
{
namespace core
{

EHM2::EHM2()
{
}

EHM2NetPtr EHM2::constructNet(const Eigen::MatrixXi& validation_matrix)
{
    int num_tracks = validation_matrix.rows();

    // Construct tree
    EHM2TreePtr tree = EHM2::constructTree(validation_matrix);

    // Initialise net
    EHM2NetNodePtr root = std::make_shared<EHM2NetNode>(EHM2NetNode(0, {}, 0, 0));
    EHM2NetPtr net = std::make_shared<EHM2Net>(EHM2Net(root, validation_matrix));

    // Construct net
    std::stack<std::pair<EHM2TreePtr, int>> stack;
    stack.push(std::make_pair(tree, 1));
    while (!stack.empty()) {
        std::pair<EHM2TreePtr, int> pair = stack.top();
        stack.pop();
        EHM2TreePtr tree = pair.first;
        int layer = pair.second;

        // Get list of nodes in previous layer of subtree
        std::set<EHM2NetNodePtr> parent_nodes = net->getNodesPerLayerSubnet(layer - 1, tree->subtree);

        // Get indices of hypothesised detections for the track
        int num_detections = net->validation_matrix.cols();
        std::set<int> v_detections;
        for (int detection = 0; detection < num_detections; detection++)
        {
            if (net->validation_matrix(tree->track, detection) == 1)
            {
                v_detections.insert(detection);
            }
        }

        // If this is not a leaf layer
        if (tree->children.size() > 0) {
            // Process each subtree
            for (auto& child_tree : tree->children) {

                // Compute accumulated measurements up to next layer (i+1)
                std::set<int> acc(child_tree->detections.begin(), child_tree->detections.end());
                acc.insert(0);

                // List of nodes in current layer
                std::map<std::vector<int>, std::set<EHM2NetNodePtr>> children_per_identity;

                // For each parent node
                for (auto& parent : parent_nodes) {

                    // Exclude any detections already considered by parent nodes (always include null)
                    std::set<int> v_detections_m1;
                    std::set_difference(v_detections.begin(), v_detections.end(),
                        parent->identity.begin(), parent->identity.end(),
                        std::inserter(v_detections_m1, v_detections_m1.begin()));
                    v_detections_m1.insert(0);

                    // Iterate over valid detections
                    for (int j : v_detections_m1)
                    {
                        // Identity
                        // identity = acc.intersection(parent.identity | {j}) - {0}
                        EHMNetNodeIdentity identity;
                        std::set<int> inter(parent->identity);
                        inter.insert(j);
                        std::set_intersection(acc.begin(), acc.end(), inter.begin(), inter.end(), std::inserter(identity, identity.begin()));
                        identity.erase(0);

                        // Find valid nodes in current layer that have the same identity
                        std::vector<int> identity_vec(identity.begin(), identity.end());

                        if (children_per_identity.find(identity_vec) == children_per_identity.end()) {
                            // Create new node
                            EHM2NetNodePtr child = std::make_shared<EHM2NetNode>(EHM2NetNode(layer, identity, child_tree->track, child_tree->subtree));
                            // Add node to net
                            net->addNode(child, parent, j);
                            // Add node to list of children for the identity
                            children_per_identity[identity_vec].insert(child);
                        }
                        else {
                            const std::set<EHM2NetNodePtr>& v_children = children_per_identity[identity_vec];
                            // Simply add new edge or update existing one
                            for (EHM2NetNodePtr child : v_children) {
                                net->addEdge(parent, child, j);
                            }
                        }
                    }
                }
            }
        }
        else {
            // For all nodes in previous layer
            for (auto& parent : parent_nodes) {

                // Exclude any detections already considered by parent nodes (always include null)
                std::set<int> v_detections_m1;
                std::set_difference(v_detections.begin(), v_detections.end(),
                    parent->identity.begin(), parent->identity.end(),
                    std::inserter(v_detections_m1, v_detections_m1.begin()));
                v_detections_m1.insert(0);

                // Get leaf children
                std::set<EHM2NetNodePtr> leaves = net->getNodesPerLayerSubnet(layer, tree->subtree);
                EHM2NetNodePtr child;
                if (leaves.size() > 0)
                    child = *leaves.begin();

                // For each valid detection
                for (int j : v_detections_m1) {
                    // If layer is empty or no valid node exist, add new node
                    if (child == nullptr) {
                        // Create new node
                        child = std::make_shared<EHM2NetNode>(EHM2NetNode(layer, {}, -1, tree->subtree));
                        // Add node to net
                        net->addNode(child, parent, j);
                    }
                    else {
                        // Simply add new edge or update existing one
                        net->addEdge(parent, child, j);
                    }
                }
            }
        }

        // Recursively process children
        for (auto& child_tree : tree->children) {
            stack.push(std::make_pair(child_tree, layer + 1));
        }
    }

    // Compute and cache nodes per track
    for (int i = 0; i < num_tracks; i++) {
        std::set<EHM2NetNodePtr> nodes_per_track;
        for (EHM2NetNodePtr node : net->getNodes()) {
            if (node->track == i) {
                nodes_per_track.insert(node);
            }
        }
        net->nodes_per_track[i] = nodes_per_track;
    }

    
    return net;

}

Eigen::MatrixXd EHM2::computeAssociationMatrix(const EHM2NetPtr net, const Eigen::MatrixXd& likelihood_matrix)
{
    int num_tracks = likelihood_matrix.rows();
    int num_detections = likelihood_matrix.cols();
    int num_nodes = net->getNumNodes();
    std::vector<EHM2NetNodePtr> nodes = net->getNodes();
    std::vector<EHM2NetNodePtr> nodes_forward = net->getNodesForward();

    // Precompute valid detections per track
    std::vector<std::set<int>> valid_detections_per_track;
    for (int i = 0; i < num_tracks; i++) {
        std::set<int> v_detections;
        for (int detection = 0; detection < num_detections; detection++) {
            if (net->validation_matrix(i, detection) == 1) {
                v_detections.insert(detection);
            }
        }
        valid_detections_per_track.push_back(v_detections);
    }

    // Compute w_B (Backward-pass) - Eq. (47) of [EHM2]
    Eigen::VectorXd w_B = Eigen::VectorXd::Zero(num_nodes);
    for (int i = num_nodes - 1; i >= 0; i--) {
        EHM2NetNodePtr parent = nodes_forward[i];
        int p_i = parent->id;
        if (parent->track == -1) {
            w_B(p_i) = 1;
        }
        else {
            double weight = 0;
            std::set<int> v_detections_tmp = valid_detections_per_track[parent->track];
            std::set<int> v_detections;
            std::set_difference(v_detections_tmp.begin(), v_detections_tmp.end(), parent->identity.begin(), parent->identity.end(), 
                                std::inserter(v_detections, v_detections.begin()));
            for (int detection : v_detections) {
                std::vector<EHM2NetNodePtr> v_children = net->getChildrenPerDetection(parent, detection);
                double weight_det = likelihood_matrix(parent->track, detection);
                for (EHM2NetNodePtr child : v_children) {
                    int c_i = child->id;
                    weight_det *= w_B(c_i);
                }
                weight += weight_det;
            }
            w_B(p_i) = weight;
        }
    }

    // Compute w_F (Forward-pass) - Eq. (49) of [EHM2]
    Eigen::VectorXd w_F = Eigen::VectorXd::Zero(num_nodes);
    w_F(0) = 1;
    for (auto& parent: nodes_forward) {
        if (parent->track == -1) {
            continue;
        }
        int p_i = parent->id;
        std::set<int> v_detections_tmp = valid_detections_per_track[parent->track];
        std::set<int> v_detections;
        std::set_difference(v_detections_tmp.begin(), v_detections_tmp.end(), parent->identity.begin(), parent->identity.end(), 
                            std::inserter(v_detections, v_detections.begin()));
        for (int detection : v_detections) {
            const std::vector<EHM2NetNodePtr>& v_children = net->getChildrenPerDetection(parent, detection);
            std::set<int> v_children_inds;
            for (EHM2NetNodePtr child : v_children) {
                v_children_inds.insert(child->id);
            }
            for (EHM2NetNodePtr child : v_children) {
                if (child->track == -1) {
                    continue;
                }
                int c_i = child->id;
                std::set<int> sibling_inds(v_children_inds.begin(), v_children_inds.end());
                sibling_inds.erase(c_i);
                double sibling_weight = 1;
                for (int sibling_ind : sibling_inds) {
                    sibling_weight *= w_B(sibling_ind);
                }
                w_F(c_i) += w_F(p_i) * likelihood_matrix(parent->track, detection) * sibling_weight;
            }
        }
    }
    // Compute association probs - Eq. (46) of [EHM2]
    Eigen::MatrixXd a_matrix = Eigen::MatrixXd::Zero(num_tracks, num_detections);
    for (int track = 0; track < num_tracks; track++) {
        std::set<int> v_detections = valid_detections_per_track[track];
        for (int detection : v_detections) {
            for (auto& parent : net->nodes_per_track[track]) {
                std::vector<EHM2NetNodePtr> v_children = net->getChildrenPerDetection(parent, detection);
                if (v_children.empty()) {
                    continue;
                }
                double weight = likelihood_matrix(track, detection) * w_F(parent->id);
                for (auto& child : v_children) {
                    weight *= w_B(child->id);
                }
                a_matrix(track, detection) += weight;
            }
        }
        // Normalise
        a_matrix(track, Eigen::all) /= a_matrix(track, Eigen::all).sum();
    }

    return a_matrix;
}

Eigen::MatrixXd EHM2::run(const Eigen::MatrixXi& validation_matrix, const Eigen::MatrixXd& likelihood_matrix)
{
    // Cluster tracks into groups that share common detections
    std::vector<ClusterPtr> clusters = genClusters(validation_matrix, likelihood_matrix);

    // Initialise the association probabilities matrix
    Eigen::MatrixXd a_matrix = Eigen::MatrixXd::Zero(validation_matrix.rows(), validation_matrix.cols());

    // Perform EHM for each cluster
    for (ClusterPtr cluster : clusters) {
        // Extract track and detection indices
        std::vector<int> c_tracks = cluster->tracks;
        std::vector<int> c_detections = cluster->detections;

        if (c_detections.size() == 0) {
            a_matrix(c_tracks, 0).setOnes();
            continue;
        }

        // Extract validation and likelihood matrices for cluster
        Eigen::MatrixXi c_validation_matrix = cluster->validation_matrix;
        Eigen::MatrixXd c_likelihood_matrix = cluster->likelihood_matrix;

        // Construct the EHM net
        EHM2NetPtr net = constructNet(c_validation_matrix);

        // Compute the association probabilities
        Eigen::MatrixXd c_a_matrix = computeAssociationMatrix(net, c_likelihood_matrix);

        // Update the association probabilities matrix
        a_matrix(c_tracks, c_detections) = c_a_matrix;
    }
    return a_matrix;
}

EHM2TreePtr EHM2::constructTree(const Eigen::MatrixXi& validation_matrix)
{
    int num_tracks = validation_matrix.rows();
    int num_detections = validation_matrix.cols();

    std::vector<EHM2TreePtr> trees;
    int last_subtree_index = -1;

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
            if (intersection.size() > 0) {
                // Add the track to the tree
                matched.push_back(j);
            }
            else {
                unmatched.push_back(j);
            }
        }

        std::vector<EHM2TreePtr> children;
        if (matched.size()) {
            for (int j = 0; j < matched.size(); j++) {
                children.push_back(trees[matched[j]]);
            }
            std::set<int> detections(v_detections.begin(), v_detections.end());
            for (auto& tree : children) {
                std::set_union(detections.begin(), detections.end(), tree->detections.begin(), tree->detections.end(), std::inserter(detections, detections.begin()));
            }
            std::vector<int> subtree_indices;
            for (auto& tree : children) {
                subtree_indices.push_back(tree->subtree);
            }
            int subtree_index = *std::max_element(subtree_indices.begin(), subtree_indices.end());
            EHM2TreePtr tree = std::make_shared<EHM2Tree>(EHM2Tree(i, children, detections, subtree_index));

            std::vector<EHM2TreePtr> new_trees;
            for (auto& j : unmatched) {
                new_trees.push_back(trees[j]);
            }
            new_trees.push_back(tree);
            trees = new_trees;
        }
        else {
            last_subtree_index += 1;
            EHM2TreePtr tree = std::make_shared<EHM2Tree>(EHM2Tree(i, children, v_detections, last_subtree_index));
            trees.push_back(tree);
        }
    }

    // TODO: Add error if trees.size() != 1

    EHM2TreePtr tree = trees[0];

    // Reverse subtree indices
    int max_subtree_ind = tree->subtree;
    std::vector<EHM2TreePtr> nodes = tree->getAllChildren();
    tree->subtree = 0;
    for (auto& node : nodes) {
        node->subtree = max_subtree_ind - node->subtree;
    }
    return tree;
}

} // namespace core
} // namespace ehm
