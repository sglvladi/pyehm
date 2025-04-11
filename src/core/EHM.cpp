#include "EHM.h"


namespace ehm
{
namespace core
{

EHM::EHM()
{
}

EHMNetPtr EHM::constructNet(const Eigen::MatrixXi& validation_matrix)
{
	// Cache number of tracks and detections
    int num_tracks = validation_matrix.rows();
    int num_detections = validation_matrix.cols();

    // Construct tree
    EHMTreePtr tree = constructTree(validation_matrix);
    std::vector<EHMTreePtr> nodes = tree->getNodes();

    // Initialise net
    EHMNetNodePtr root = std::make_shared<EHMNetNode>(EHMNetNode(0, {}));
    EHMNetPtr net = std::make_shared<EHMNet>(EHMNet(root, validation_matrix, tree));

    // Construct net
    std::stack<EHMTreePtr> stack;
    stack.push(tree);
    while (!stack.empty()) {
        EHMTreePtr tree = stack.top();
		stack.pop();

        // Get list of nodes in previous layer of subtree
        std::vector<EHMNetNodePtr> parent_nodes = net->getNodesByLayer(tree->track);

        // Get indices of hypothesised detections for the track
        std::set<int> v_detections;
        for (int detection = 0; detection < num_detections; detection++)
        {
            if (net->validation_matrix(tree->track, detection) == 1)
            {
                v_detections.insert(detection);
            }
        }

        // If this is not a leaf layer
        if (!tree->children.empty()) {
            // Process each subtree
            for (auto& child_tree : tree->children) {

                // Compute accumulated measurements up to next layer (i+1)
                std::set<int> acc(child_tree->detections.begin(), child_tree->detections.end());

                // List of nodes in current layer
                std::map<std::vector<int>, std::set<EHMNetNodePtr>> children_per_identity;

                // For each parent node
                for (auto& parent : parent_nodes) {

                    // Exclude any detections already considered by parent nodes (always include null)
					std::unordered_set<int> v_detections_m1 = computeRemainingDetections(v_detections, parent->identity);

                    // Iterate over valid detections
                    for (int j : v_detections_m1)
                    {
                        // Identity
						EHMNetNodeIdentity identity = computeIdentity(acc, parent->identity, j);

                        // Find valid nodes in current layer that have the same identity
                        std::vector<int> identity_vec(identity.begin(), identity.end());

                        if (children_per_identity.find(identity_vec) == children_per_identity.end()) {
                            // Create new node
                            EHMNetNodePtr child = std::make_shared<EHMNetNode>(EHMNetNode(child_tree->track, identity));
                            // Add node to net
                            net->addNode(child, parent, j);
                            // Add node to list of children for the identity
                            children_per_identity[identity_vec].insert(child);
                        }
                        else {
                            const std::set<EHMNetNodePtr>& v_children = children_per_identity[identity_vec];
                            // Simply add new edge or update existing one
                            for (EHMNetNodePtr child : v_children) {
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
				std::unordered_set<int> v_detections_m1 = computeRemainingDetections(v_detections, parent->identity);

                // Get leaf children
                std::vector<EHMNetNodePtr> leaves = net->getNodesByLayer(-1);
                EHMNetNodePtr child = leaves.empty() ? nullptr : *leaves.begin();

                // For each valid detection
                for (int j : v_detections_m1) {
                    // If layer is empty or no valid node exist, add new node
                    if (child == nullptr) {
                        // Create new node
                        child = std::make_shared<EHMNetNode>(EHMNetNode(-1));
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
            stack.push(child_tree);
        }
    }

    return net;

}

Eigen::MatrixXd EHM::computeAssociationMatrix(const EHMNetPtr net, const Eigen::MatrixXd& likelihood_matrix)
{
    int num_tracks = likelihood_matrix.rows();
    int num_detections = likelihood_matrix.cols();
    int num_nodes = net->getNumNodes();
    std::vector<EHMNetNodePtr> nodes = net->getNodes();
    std::vector<EHMNetNodePtr> nodes_forward = net->getNodesForward();

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
        EHMNetNodePtr parent = nodes_forward[i];
        int p_i = parent->id;
        if (parent->layer == -1) {
            w_B(p_i) = 1;
        }
        else {
            double weight = 0;
            std::set<int> v_detections_tmp = valid_detections_per_track[parent->layer];
			std::unordered_set<int> v_detections = computeRemainingDetections(v_detections_tmp, parent->identity);
            for (int detection : v_detections) {
                std::vector<EHMNetNodePtr> v_children = net->getChildrenPerDetection(parent, detection);
                double weight_det = likelihood_matrix(parent->layer, detection);
                for (EHMNetNodePtr child : v_children) {
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
    for (auto& parent : nodes_forward) {
        if (parent->layer == -1) {
            continue;
        }
        int p_i = parent->id;
        std::set<int> v_detections_tmp = valid_detections_per_track[parent->layer];
		std::unordered_set<int> v_detections = computeRemainingDetections(v_detections_tmp, parent->identity);
        for (int detection : v_detections) {
            const std::vector<EHMNetNodePtr>& v_children = net->getChildrenPerDetection(parent, detection);
            std::set<int> v_children_inds;
            for (EHMNetNodePtr child : v_children) {
                v_children_inds.insert(child->id);
            }
            for (EHMNetNodePtr child : v_children) {
                if (child->layer == -1) {
                    continue;
                }
                int c_i = child->id;
                std::set<int> sibling_inds(v_children_inds.begin(), v_children_inds.end());
                sibling_inds.erase(c_i);
                double sibling_weight = 1;
                for (int sibling_ind : sibling_inds) {
                    sibling_weight *= w_B(sibling_ind);
                }
                w_F(c_i) += w_F(p_i) * likelihood_matrix(parent->layer, detection) * sibling_weight;
            }
        }
    }
    // Compute association probs - Eq. (46) of [EHM2]
    Eigen::MatrixXd a_matrix = Eigen::MatrixXd::Zero(num_tracks, num_detections);
    for (int track = 0; track < num_tracks; track++) {
        std::set<int> v_detections_tmp = valid_detections_per_track[track];
        for (auto& parent : net->getNodesByLayer(track)) {
            std::set<int> v_detections_tmp = valid_detections_per_track[parent->layer];
			std::unordered_set<int> v_detections = computeRemainingDetections(v_detections_tmp, parent->identity);
            for (int detection : v_detections) {
                std::vector<EHMNetNodePtr> v_children = net->getChildrenPerDetection(parent, detection);
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

Eigen::MatrixXd EHM::run(const Eigen::MatrixXi& validation_matrix, const Eigen::MatrixXd& likelihood_matrix)
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
        EHMNetPtr net = constructNet(c_validation_matrix);

        // Compute the association probabilities
        Eigen::MatrixXd c_a_matrix = computeAssociationMatrix(net, c_likelihood_matrix);

        // Update the association probabilities matrix
        a_matrix(c_tracks, c_detections) = c_a_matrix;
    }
    return a_matrix;
}

EHMTreePtr EHM::constructTree(const Eigen::MatrixXi& validation_matrix)
{
    int num_tracks = validation_matrix.rows();
    int num_detections = validation_matrix.cols();

    EHMTreePtr tree;

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

		if (tree == nullptr) {
			tree = std::make_shared<EHMTree>(EHMTree(i, {}, v_detections));
		}
        else {
            std::set<int> detections(v_detections.begin(), v_detections.end());
            std::set_union(detections.begin(), detections.end(), tree->detections.begin(), tree->detections.end(), std::inserter(detections, detections.begin()));
			tree = std::make_shared<EHMTree>(EHMTree(i, {tree}, detections));
        }
        
    }
    return tree;
}

} // namespace core
} // namespace ehm
