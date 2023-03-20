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
    // Initialise net
    EHMNetNodePtr root = std::make_shared<EHMNetNode>(EHMNetNode(-1));
    EHMNetPtr net = std::make_shared<EHMNet>(EHMNet(root, validation_matrix));

    // A layer in the network is created for each track (not counting the root-node layer)
    int num_layers = validation_matrix.rows();
    int num_detections = validation_matrix.cols();
    for (int layer = 0; layer < num_layers; layer++)
    {
        // Get list of nodes in previous layer
        std::vector<EHMNetNodePtr> parent_nodes;
        for (EHMNetNodePtr node : net->getNodes()) {
            if (node->layer == layer - 1) {
                parent_nodes.push_back(node);
            }
        }

        // Get indices of hypothesised detections for the track
        std::set<int> v_detections;
        for (int detection = 0; detection < num_detections; detection++)
        {   
            if (validation_matrix(layer, detection) == 1)
            {
                v_detections.insert(detection);
			}
		}
        
        // Compute accumulated measurements up to next layer (i+1)
        std::set<int> acc = {};
        for (int i = layer + 1; i < num_layers; i++)
        {
            for (int detection = 0; detection < num_detections; detection++)
            {
                if (validation_matrix(i, detection) == 1)
                {
					acc.insert(detection);
				}
			}
		}

        // List of nodes in current layer
        std::map<std::vector<int>, std::set<EHMNetNodePtr>> children_per_identity;
        for (EHMNetNodePtr parent : parent_nodes)
        {
            // Exclude any detections already considered by parent nodes (always include null)
            std::set<int> v_detections_m1;
            std::set_difference(v_detections.begin(), v_detections.end(), 
                                parent->identity.begin(), parent->identity.end(), 
                                std::inserter(v_detections_m1, v_detections_m1.begin()));
            v_detections_m1.insert(0);

            // Iterate over all possible detections
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
                std::set<EHMNetNodePtr> v_children = children_per_identity[identity_vec];

                if (v_children.size() == 0) {
                    // Create new node
                    EHMNetNodePtr child = std::make_shared<EHMNetNode>(EHMNetNode(layer, identity));
                    // Add node to net
                    net->addNode(child, parent, j);
                    // Add node to list of children for the identity
                    children_per_identity[identity_vec].insert(child);
                }
                else {
                    // Simply add new edge or update existing one
                    for (auto& child : v_children) {
                        net->addEdge(parent, child, j);
                    }
                }

			}
		}
	}
    return net;

}

Eigen::MatrixXd EHM::computeAssociationMatrix(const EHMNetPtr net, const Eigen::MatrixXd& likelihood_matrix)
{
    int num_tracks = likelihood_matrix.rows();
    int num_detections = likelihood_matrix.cols();
    int num_nodes = net->getNumNodes();
    std::vector<EHMNetNodePtr> net_nodes = net->getNodes();
    std::vector<int> all_detections(num_detections);
    std::iota(all_detections.begin(), all_detections.end(), 0);

    // Compute p_D (Downward-pass) - Eq. (22) of [EHM1]
    Eigen::VectorXd p_D = Eigen::VectorXd::Zero(num_nodes);
    p_D(0) = 1;
    // Iterate over all, but the first net node
    for (int i = 1; i < num_nodes; i++) {
        EHMNetNodePtr child = net_nodes[i];
        int c_i = child->id;
        std::set<EHMNetNodePtr> parents = net->getParents(child);
        for (EHMNetNodePtr parent : parents) {
            int p_i = parent->id;
            std::set<int> detections = net->getEdges(parent, child);
            std::vector<int>detections_vec(detections.begin(), detections.end());
            p_D(c_i) += (likelihood_matrix(child->layer, detections_vec) * p_D(p_i)).sum();
        }
    }

    // Compute p_U (Upward-pass) - Eq. (23) of [EHM1]
    Eigen::VectorXd p_U = Eigen::VectorXd::Zero(num_nodes);
    p_U(num_nodes-1) = 1;
    for (int i = num_nodes - 2; i >= 0; i--) {
        EHMNetNodePtr parent = net_nodes[i];
        int p_i = parent->id;
        std::set<EHMNetNodePtr> children = net->getChildren(parent);
        for (EHMNetNodePtr child : children) {
            int c_i = child->id;
            std::set<int> detections = net->getEdges(parent, child);
            std::vector<int>detections_vec(detections.begin(), detections.end());
            p_U(p_i) += (likelihood_matrix(child->layer, detections_vec) * p_U(c_i)).sum();
        }
    }


    // Compute p_DT - Eq. (21) of [EHM1]
    Eigen::MatrixXd p_DT = Eigen::MatrixXd::Zero(num_detections, num_nodes);
    for (EHMNetNodePtr child : net_nodes) {
        int c_i = child->id;
        std::set<EHMNetNodePtr> parents = net->getParents(child);
        for (EHMNetNodePtr parent : parents) {
            int p_i = parent->id;
            std::set<int> detections = net->getEdges(parent, child);
            std::vector<int>detections_vec(detections.begin(), detections.end());
            p_DT(detections_vec, c_i) = p_DT(detections_vec, c_i).array() + p_D(p_i);
        }
    }

    // Compute p_T - Eq. (20) of [EHM1]
    Eigen::MatrixXd p_T = Eigen::MatrixXd::Ones(num_detections, num_nodes);
    p_T.col(0).setZero();
    for (int i = 1; i < num_nodes; i++) {
        EHMNetNodePtr node = net_nodes[i];
        int n_i = node->id;
        p_T(Eigen::all, n_i) = p_U(n_i) * likelihood_matrix(node->layer, Eigen::all).transpose().cwiseProduct(p_DT(Eigen::all, n_i));
       /* for (int j = 0; j < num_detections; j++) {
            p_T(j, n_i) = p_U(n_i) * likelihood_matrix(node->layer, j) * p_DT(j, n_i);
        }
        int b = 2;*/
    }

    // Compute association weights - Eq. (15) of [EHM1]
    Eigen::MatrixXd a_matrix = Eigen::MatrixXd::Ones(num_tracks, num_detections);
    for (int i = 0; i < num_tracks; i++) {
        std::vector<int> node_inds;
        for (int node_ind = 0; node_ind < net_nodes.size(); node_ind++) {
            if (net_nodes[node_ind]->layer == i)
                node_inds.push_back(node_ind);
        }
        for (int j = 0; j < num_detections; j++) {
            a_matrix(i, j) = p_T(j, node_inds).sum();
        }
        // Normalise
        a_matrix(i, Eigen::all) /= a_matrix(i, Eigen::all).sum();
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

} // namespace core
} // namespace ehm