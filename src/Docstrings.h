#pragma once
#include <string>

namespace docstrings
{

    std::string EHMNetNode();

    std::string EHMNet();
    std::string EHMNet___init__();
    std::string EHMNet_num_layers();
    std::string EHMNet_num_nodes();
    std::string EHMNet_root();
    std::string EHMNet_nodes();
    std::string EHMNet_nodes_forward();
    std::string EHMNet_add_node();
    std::string EHMNet_add_edge();

    std::string EHMTree();
    std::string EHMTree_depth();
    std::string EHMTree_nodes();

    std::string Cluster();
    std::string gen_clusters_log_likelihood_matrix();
    std::string gen_clusters_track_hypotheses();

    std::string EHM();
    std::string EHM_construct_net();
    std::string EHM_construct_tree();
    std::string EHM_compute_association_probabilities();
    std::string EHM_run_log_likelihood_matrix();
    std::string EHM_run_track_hypotheses();

    std::string EHM2();
    std::string EHM2_construct_net();
    std::string EHM2_construct_tree();
    std::string EHM2_compute_association_probabilities();
    std::string EHM2_run_log_likelihood_matrix();
    std::string EHM2_run_track_hypotheses();



}
