#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Docstrings.h>
#include <core/EHM.h>
#include <core/EHM2.h>
#include <utils/Cluster.h>
#include <utils/Utils.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;
using namespace ehm::core;
using namespace ehm::utils;
using namespace ehm::net;


PYBIND11_MODULE(_pyehm, m) {

    py::options options;
    options.disable_function_signatures();

    // Submodules
    auto utils_m = m.def_submodule("utils");
    auto net_m = m.def_submodule("net");
    auto core_m = m.def_submodule("core");

    // Nodes
    py::class_<EHMNetNode, EHMNetNodePtr>(net_m, "EHMNetNode", docstrings::EHMNetNode().c_str())
        .def(py::init<int, EHMNetNodeIdentity>(), "layer"_a, "identity"_a = EHMNetNodeIdentity())
        .def_readwrite("id", &EHMNetNode::id)
        .def_readwrite("layer", &EHMNetNode::layer)
        .def_readwrite("identity", &EHMNetNode::identity)
        .def("__str__", &EHMNetNode::toString)
        .def("__repr__", &EHMNetNode::toString);

    // Nets
    py::class_<EHMNet, EHMNetPtr>(net_m, "EHMNet", docstrings::EHMNet().c_str())
        .def(py::init<const EHMNetNodePtr, const Eigen::MatrixXi&, const EHM2TreePtr>(), "root"_a, "validation_matrix"_a, "tree"_a, docstrings::EHMNet___init__().c_str())
        .def_readonly("validation_matrix", &EHMNet::validation_matrix)
        .def_property_readonly("num_layers", &EHMNet::getNumLayers, docstrings::EHMNet_num_layers().c_str())
        .def_property_readonly("num_nodes", &EHMNet::getNumNodes, docstrings::EHMNet_num_nodes().c_str())
        .def_property_readonly("root", &EHMNet::getRoot, docstrings::EHMNet_root().c_str())
        .def_property_readonly("nodes", &EHMNet::getNodes, docstrings::EHMNet_nodes().c_str())
        .def_property_readonly("nodes_forward", &EHMNet::getNodesForward, docstrings::EHMNet_nodes_forward().c_str())
        .def("add_node", &EHMNet::addNode, "node"_a, "parent"_a, "detection"_a, docstrings::EHMNet_add_node().c_str())
        .def("add_edge", &EHMNet::addEdge, "parent"_a, "child"_a, "detection"_a, docstrings::EHMNet_add_edge().c_str());

    py::class_<EHM2Tree, EHM2TreePtr>(net_m, "EHM2Tree", docstrings::EHM2Tree().c_str())
        .def(py::init<int, std::vector<EHM2TreePtr>, EHMNetNodeIdentity>(), "track"_a, "childred"_a, "detections"_a)
        .def_readwrite("track", &EHM2Tree::track)
        .def_readwrite("children", &EHM2Tree::children)
        .def_readwrite("detections", &EHM2Tree::detections)
        .def_property_readonly("depth", &EHM2Tree::getDepth, docstrings::EHM2Tree_depth().c_str())
        .def_property_readonly("nodes", &EHM2Tree::getNodes, docstrings::EHM2Tree_nodes().c_str());

    // Utils
    py::class_<Cluster, ClusterPtr>(utils_m, "Cluster", docstrings::Cluster().c_str())
        .def(py::init<std::vector<int>, std::vector<int>, Eigen::MatrixXi, Eigen::MatrixXd>(), "tracks"_a, "detections"_a = std::vector<int>(), "validation_matrix"_a = Eigen::MatrixXi::Zero(0, 0), "likelihood_matrix"_a = Eigen::MatrixXd::Zero(0, 0))
        .def_readwrite("tracks", &Cluster::tracks)
        .def_readwrite("detections", &Cluster::detections)
        .def_readwrite("validation_matrix", &Cluster::validation_matrix)
        .def_readwrite("likelihood_matrix", &Cluster::likelihood_matrix);
    utils_m.def("gen_clusters", &genClusters, "validation_matrix"_a, "likelihood_matrix"_a = Eigen::MatrixXd::Zero(0, 0), docstrings::gen_clusters().c_str());

    // Algorithms
    py::class_<EHM>(core_m, "EHM", docstrings::EHM().c_str())
        .def(py::init<>())
        .def("construct_net", &EHM::constructNet, "validation_matrix"_a, docstrings::EHM_construct_net().c_str())
        .def("construct_tree", &EHM::constructTree, "validation_matrix"_a, docstrings::EHM2_construct_tree().c_str())
        .def("compute_association_probabilities", &EHM::computeAssociationMatrix, "net"_a, "likelihood_matrix"_a, docstrings::EHM_compute_association_probabilities().c_str())
        .def("run", &EHM::run, "validation_matrix"_a, "likelihood_matrix"_a, docstrings::EHM_run().c_str());
    py::class_<EHM2>(core_m, "EHM2", docstrings::EHM2().c_str())
        .def(py::init<>())
        .def("construct_net", &EHM2::constructNet, "validation_matrix"_a, docstrings::EHM2_construct_net().c_str())
        .def("construct_tree", &EHM2::constructTree, "validation_matrix"_a, docstrings::EHM2_construct_tree().c_str())
        .def("compute_association_probabilities", &EHM2::computeAssociationMatrix, "net"_a, "likelihood_matrix"_a, docstrings::EHM2_compute_association_probabilities().c_str())
        .def("run", &EHM2::run, "validation_matrix"_a, "likelihood_matrix"_a, docstrings::EHM2_run().c_str());

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}