#pragma once
#include <string>

namespace docstrings
{

    std::string EHMNetNode();
    std::string EHM2NetNode();

    std::string EHMNet();
    std::string EHMNet___init__();
    std::string EHMNet_num_layers();
    std::string EHMNet_num_nodes();
    std::string EHMNet_root();
    std::string EHMNet_nodes();
    std::string EHMNet_nodes_forward();
    std::string EHMNet_get_parents();
    std::string EHMNet_get_children();
    std::string EHMNet_get_edges();
    std::string EHMNet_add_node();
    std::string EHMNet_add_edge();

    std::string EHM2Net();
    std::string EHM2Net_num_layers();
    std::string EHM2Net_num_nodes();
    std::string EHM2Net_root();
    std::string EHM2Net_nodes();
    std::string EHM2Net_nodes_forward();
    std::string EHM2Net_nodes_per_track();
    std::string EHM2Net_add_node();
    std::string EHM2Net_add_edge();
    std::string EHM2Net_get_nodes_per_layer_subnet();
    std::string EHM2Net_get_children_per_detection();

    std::string EHM2Tree();
    std::string EHM2Tree_depth();

    std::string Cluster();
    std::string gen_clusters();

    std::string EHM();
    std::string EHM_construct_net();
    std::string EHM_compute_association_probabilities();
    std::string EHM_run();

    std::string EHM2();
    std::string EHM2_construct_net();
    std::string EHM2_construct_tree();
    std::string EHM2_compute_association_probabilities();
    std::string EHM2_run();



}
