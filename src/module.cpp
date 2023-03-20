#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

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


PYBIND11_MODULE(_pyehm, m) {

    // Submodules
	auto utils_m = m.def_submodule("utils");
	auto core_m = m.def_submodule("core");


    // Class definitions
    auto pyEHMNetNode = py::class_<EHMNetNode, EHMNetNodePtr>(utils_m, "EHMNetNode", docstrings::EHMNetNode().c_str());
    auto pyEHM2NetNode = py::class_<EHM2NetNode, EHM2NetNodePtr>(utils_m, "EHM2NetNode", docstrings::EHM2NetNode().c_str());

	// Nodes
	pyEHMNetNode
		.def(py::init<int, EHMNetNodeIdentity>(), "layer"_a, "identity"_a = EHMNetNodeIdentity())
		.def_readwrite("id", &EHMNetNode::id)
		.def_readwrite("layer", &EHMNetNode::layer)
		.def_readwrite("identity", &EHMNetNode::identity)
		.def("__str__", &EHMNetNode::toString)
		.def("__repr__", &EHMNetNode::toString);
	pyEHM2NetNode
		.def(py::init<int, int, int, EHMNetNodeIdentity>(), "layer"_a, "track"_a = -1, "subnet"_a = 0, "identity"_a = EHMNetNodeIdentity())
		.def_readwrite("id", &EHM2NetNode::id)
		.def_readwrite("layer", &EHM2NetNode::layer)
		.def_readwrite("identity", &EHM2NetNode::identity)
		.def_readwrite("track", &EHM2NetNode::track)
		.def_readwrite("subnet", &EHM2NetNode::subnet)
		.def("__str__", &EHM2NetNode::toString)
		.def("__repr__", &EHM2NetNode::toString);

	// Nets
	py::class_<EHMNet, EHMNetPtr>(utils_m, "EHMNet", docstrings::EHMNet().c_str())
		.def(py::init<const EHMNetNodePtr, const Eigen::MatrixXi&>(), "root"_a, "validation_matrix"_a)
		.def_readonly("validation_matrix", &EHMNet::validation_matrix)
		.def_property_readonly("num_layers", &EHMNet::getNumLayers, docstrings::EHMNet_num_layers().c_str())
		.def_property_readonly("num_nodes", &EHMNet::getNumNodes, docstrings::EHMNet_num_nodes().c_str())
		.def_property_readonly("root", &EHMNet::getRoot, docstrings::EHMNet_root().c_str())
		.def_property_readonly("nodes", &EHMNet::getNodes, docstrings::EHMNet_nodes().c_str())
		.def_property_readonly("nodes_forward", &EHMNet::getNodesForward, docstrings::EHMNet_nodes_forward().c_str())
		.def("get_parents", &EHMNet::getParents, docstrings::EHMNet_get_parents().c_str())
		.def("get_children", &EHMNet::getChildren, docstrings::EHMNet_get_children().c_str())
		.def("get_edges", &EHMNet::getEdges, docstrings::EHMNet_get_edges().c_str())
		.def("add_node", &EHMNet::addNode, docstrings::EHMNet_add_node().c_str())
		.def("add_edge", &EHMNet::addEdge, docstrings::EHMNet_add_edge().c_str());

	py::class_<EHM2Net, EHM2NetPtr>(utils_m, "EHM2Net")
		.def(py::init<const EHM2NetNodePtr, const Eigen::MatrixXi&>(), "root"_a, "validation_matrix"_a)
		.def_readonly("validation_matrix", &EHM2Net::validation_matrix)
		.def_property_readonly("num_layers", &EHM2Net::getNumLayers, docstrings::EHM2Net_num_layers().c_str())
		.def_property_readonly("num_nodes", &EHM2Net::getNumNodes, docstrings::EHM2Net_num_nodes().c_str())
		.def_property_readonly("root", &EHM2Net::getRoot, docstrings::EHM2Net_root().c_str())
		.def_property_readonly("nodes", &EHM2Net::getNodes, docstrings::EHM2Net_nodes().c_str())
		.def_property_readonly("nodes_forward", &EHM2Net::getNodesForward, docstrings::EHM2Net_nodes_forward().c_str())
		.def_readonly("nodes_per_track", &EHM2Net::nodes_per_track)
		.def("get_nodes_per_layer_subnet", &EHM2Net::getNodesPerLayerSubnet, docstrings::EHM2Net_get_nodes_per_layer_subnet().c_str())
		.def("get_children_per_detection", &EHM2Net::getChildrenPerDetection)
		.def("add_node", &EHM2Net::addNode, docstrings::EHM2Net_add_node().c_str())
		.def("add_edge", &EHM2Net::addEdge, docstrings::EHM2Net_add_edge().c_str());

	// Utils
	py::class_<EHM2Tree, EHM2TreePtr>(utils_m, "EHM2Tree")
		.def(py::init<int, std::vector<EHM2TreePtr>, EHMNetNodeIdentity, int>(), "track"_a, "childred"_a, "detections"_a, "subtree"_a)
		.def_readwrite("track", &EHM2Tree::track)
		.def_readwrite("children", &EHM2Tree::children)
		.def_readwrite("detections", &EHM2Tree::detections)
		.def_readwrite("subtree", &EHM2Tree::subtree)
		.def_property_readonly("depth", &EHM2Tree::getDepth);
	py::class_<Cluster, ClusterPtr>(utils_m, "Cluster")
		.def(py::init<std::vector<int>, std::vector<int>, Eigen::MatrixXi, Eigen::MatrixXd>(), "tracks"_a, "detections"_a, "validation_matrix"_a = Eigen::MatrixXi::Zero(0, 0), "likelihood_matrix"_a = Eigen::MatrixXd::Zero(0, 0))
		.def_readwrite("tracks", &Cluster::tracks)
		.def_readwrite("detections", &Cluster::detections)
		.def_readwrite("validation_matrix", &Cluster::validation_matrix)
		.def_readwrite("likelihood_matrix", &Cluster::likelihood_matrix);
	utils_m.def("gen_clusters", &genClusters, "validation_matrix"_a, "likelihood_matrix"_a);

	// Algorithms
	py::class_<EHM>(core_m, "EHM")
		.def(py::init<>())
		.def_static("construct_net", &EHM::constructNet, "validation_matrix"_a)
		.def_static("compute_association_probabilities", &EHM::computeAssociationMatrix, "net"_a, "likelihood_matrix"_a)
		.def_static("run", &EHM::run, "validation_matrix"_a, "likelihood_matrix"_a);
	py::class_<EHM2>(core_m, "EHM2")
		.def(py::init<>())
		.def_static("construct_net", &EHM2::constructNet, "validation_matrix"_a)
		.def_static("construct_tree", &EHM2::constructTree, "validation_matrix"_a)
		.def_static("compute_association_probabilities", &EHM2::computeAssociationMatrix, "net"_a, "likelihood_matrix"_a)
		.def_static("run", &EHM2::run, "validation_matrix"_a, "likelihood_matrix"_a);

#ifdef VERSION_INFO
	m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
	m.attr("__version__") = "dev";
#endif

}