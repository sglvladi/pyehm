#include "EHM.h"

#include <algorithm>
#include <stdexcept>


namespace ehm
{
namespace core
{

EHM::EHM()
{
}

EHMNetPtr EHM::constructNet(const Eigen::MatrixXd& log_likelihood_matrix)
{
	// Cache number of tracks and detections
    int num_tracks = log_likelihood_matrix.rows();
    int num_detections = log_likelihood_matrix.cols();

    // Construct tree
    EHMTreePtr tree = constructTree(log_likelihood_matrix);
    std::vector<EHMTreePtr> nodes = tree->getNodes();

    // Initialise net
    EHMNetNodePtr root = std::make_shared<EHMNetNode>(EHMNetNode(0, {}));
    EHMNetPtr net = std::make_shared<EHMNet>(EHMNet(root, log_likelihood_matrix, tree));

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
            if (net->log_likelihood_matrix(tree->track, detection) != LOG_ZERO)
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

Eigen::MatrixXd EHM::computeAssociationMatrix(const EHMNetPtr net, const Eigen::MatrixXd& log_likelihood_matrix)
{
    // NOTE: All weights are kept in log space, i.e. products become sums and sums become log-adds,
    // so that long products of small likelihoods cannot underflow.
    int num_tracks = log_likelihood_matrix.rows();
    int num_detections = log_likelihood_matrix.cols();
    int num_nodes = net->getNumNodes();
    std::vector<EHMNetNodePtr> nodes = net->getNodes();
    std::vector<EHMNetNodePtr> nodes_forward = net->getNodesForward();

    // Precompute valid detections per track
    std::vector<std::set<int>> valid_detections_per_track;
    for (int i = 0; i < num_tracks; i++) {
        std::set<int> v_detections;
        for (int detection = 0; detection < num_detections; detection++) {
            if (net->log_likelihood_matrix(i, detection) != LOG_ZERO) {
                v_detections.insert(detection);
            }
        }
        valid_detections_per_track.push_back(v_detections);
    }

    // Compute log w_B (Backward-pass) - Eq. (47) of [EHM2]
    Eigen::VectorXd log_w_B = Eigen::VectorXd::Constant(num_nodes, LOG_ZERO);
    for (int i = num_nodes - 1; i >= 0; i--) {
        EHMNetNodePtr parent = nodes_forward[i];
        int p_i = parent->id;
        if (parent->layer == -1) {
            log_w_B(p_i) = 0;
        }
        else {
            double log_weight = LOG_ZERO;
            std::set<int> v_detections_tmp = valid_detections_per_track[parent->layer];
			std::unordered_set<int> v_detections = computeRemainingDetections(v_detections_tmp, parent->identity);
            for (int detection : v_detections) {
                std::vector<EHMNetNodePtr> v_children = net->getChildrenPerDetection(parent, detection);
                double log_weight_det = log_likelihood_matrix(parent->layer, detection);
                for (EHMNetNodePtr child : v_children) {
                    int c_i = child->id;
                    log_weight_det += log_w_B(c_i);
                }
                log_weight = logAdd(log_weight, log_weight_det);
            }
            log_w_B(p_i) = log_weight;
        }
    }

    // Compute log w_F (Forward-pass) - Eq. (49) of [EHM2]
    Eigen::VectorXd log_w_F = Eigen::VectorXd::Constant(num_nodes, LOG_ZERO);
    log_w_F(0) = 0;
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
                double log_sibling_weight = 0;
                for (int sibling_ind : sibling_inds) {
                    log_sibling_weight += log_w_B(sibling_ind);
                }
                log_w_F(c_i) = logAdd(log_w_F(c_i), log_w_F(p_i) + log_likelihood_matrix(parent->layer, detection) + log_sibling_weight);
            }
        }
    }
    // Compute log association probs - Eq. (46) of [EHM2]
    Eigen::MatrixXd log_a_matrix = Eigen::MatrixXd::Constant(num_tracks, num_detections, LOG_ZERO);
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
                double log_weight = log_likelihood_matrix(track, detection) + log_w_F(parent->id);
                for (auto& child : v_children) {
                    log_weight += log_w_B(child->id);
                }
                log_a_matrix(track, detection) = logAdd(log_a_matrix(track, detection), log_weight);
            }
        }
        // Normalise
        double log_norm = LOG_ZERO;
        for (int detection = 0; detection < num_detections; detection++) {
            log_norm = logAdd(log_norm, log_a_matrix(track, detection));
        }
        log_a_matrix(track, Eigen::all).array() -= log_norm;
    }

    return log_a_matrix;
}

Eigen::MatrixXd EHM::run(const Eigen::MatrixXd& log_likelihood_matrix)
{
    TrackHypotheses hypotheses = toTrackHypotheses(log_likelihood_matrix);
    std::vector<double> log_probabilities = runHypotheses(hypotheses);

    // Write the log association probabilities to their respective matrix entries
    Eigen::MatrixXd log_a_matrix = Eigen::MatrixXd::Constant(log_likelihood_matrix.rows(), log_likelihood_matrix.cols(), LOG_ZERO);
    for (int track = 0; track < hypotheses.numTracks(); track++) {
        for (Eigen::Index k = hypotheses.offsets[track]; k < hypotheses.offsets[track + 1]; k++) {
            log_a_matrix(track, hypotheses.detections[k]) = log_probabilities[k];
        }
    }
    return log_a_matrix;
}

std::vector<Eigen::VectorXd> EHM::run(const std::vector<TrackHypothesisArrays>& track_hypotheses)
{
    TrackHypotheses hypotheses = toTrackHypotheses(track_hypotheses);
    std::vector<double> log_probabilities = runHypotheses(hypotheses);

    // Split the log association probabilities per track
    std::vector<Eigen::VectorXd> track_log_probabilities;
    track_log_probabilities.reserve(hypotheses.numTracks());
    for (int track = 0; track < hypotheses.numTracks(); track++) {
        Eigen::Index num_hypotheses = hypotheses.offsets[track + 1] - hypotheses.offsets[track];
        track_log_probabilities.push_back(Eigen::Map<const Eigen::VectorXd>(log_probabilities.data() + hypotheses.offsets[track], num_hypotheses));
    }
    return track_log_probabilities;
}

std::vector<double> EHM::runHypotheses(const TrackHypotheses& hypotheses)
{
    // Cluster tracks into groups that share common detections
    std::vector<ClusterPtr> clusters = genClusters(hypotheses);

    // Initialise the log association probabilities, one per hypothesis
    std::vector<double> log_probabilities(hypotheses.detections.size(), LOG_ZERO);

    // Perform EHM for each cluster
    for (ClusterPtr cluster : clusters) {
        // Extract track and detection indices
        const std::vector<int>& c_tracks = cluster->tracks;
        const std::vector<int>& c_detections = cluster->detections;

        if (c_detections.size() == 0) {
            // Tracks with no valid detections are missed with probability 1 (i.e. log-probability 0)
            for (int track : c_tracks) {
                for (Eigen::Index k = hypotheses.offsets[track]; k < hypotheses.offsets[track + 1]; k++) {
                    if (hypotheses.detections[k] == 0) {
                        log_probabilities[k] = 0;
                    }
                }
            }
            continue;
        }

        // Construct the EHM net
        EHMNetPtr net = constructNet(cluster->log_likelihood_matrix);

        // Compute the log association probabilities
        Eigen::MatrixXd c_log_a_matrix = computeAssociationMatrix(net, cluster->log_likelihood_matrix);

        // Write the log association probabilities of each valid hypothesis
        for (int i = 0; i < c_tracks.size(); i++) {
            int track = c_tracks[i];
            for (Eigen::Index k = hypotheses.offsets[track]; k < hypotheses.offsets[track + 1]; k++) {
                if (hypotheses.log_likelihoods[k] != LOG_ZERO) {
                    auto column = std::lower_bound(c_detections.begin(), c_detections.end(), hypotheses.detections[k]);
                    log_probabilities[k] = c_log_a_matrix(i, column - c_detections.begin());
                }
            }
        }
    }
    return log_probabilities;
}

EHMTreePtr EHM::constructTree(const Eigen::MatrixXd& log_likelihood_matrix)
{
    int num_tracks = log_likelihood_matrix.rows();
    int num_detections = log_likelihood_matrix.cols();

    EHMTreePtr tree;

    for (int i = num_tracks - 1; i >= 0; i--) {
        // Get indices of hypothesised detections for the track (minus the null hypothesis)
        std::vector<int> v_detections;
        for (int detection = 1; detection < num_detections; detection++)
        {
            if (log_likelihood_matrix(i, detection) != LOG_ZERO)
            {
                v_detections.push_back(detection);
            }
        }

		if (tree == nullptr) {
			tree = std::make_shared<EHMTree>(EHMTree(i, {}, v_detections));
		}
        else {
            // Output must not alias the inputs, as back_inserter may reallocate them
            std::vector<int> detections;
			std::set_union(v_detections.begin(), v_detections.end(), tree->detections.begin(), tree->detections.end(), std::back_inserter(detections));
			tree = std::make_shared<EHMTree>(EHMTree(i, {tree}, detections));
        }
        
    }
    if (tree == nullptr) {
        throw std::invalid_argument("Expected at least one track. Clusters generated by gen_clusters for "
                                    "tracks with no detections have no matrix and should be skipped.");
    }
    return tree;
}

} // namespace core
} // namespace ehm
