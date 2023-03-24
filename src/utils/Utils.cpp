#include "Utils.h"

namespace ehm
{
namespace utils
{

void dfs(int vertex, const std::vector<std::vector<int>>& graph, std::vector<bool>& visited, std::vector<int>& component)
{
    visited[vertex] = true;  // mark vertex as visited
    component.push_back(vertex);  // add vertex to current component

    // visit all neighbors of vertex that haven't been visited yet
    for (int neighbor = 0; neighbor < graph.size(); ++neighbor) {
        if (graph[vertex][neighbor] && !visited[neighbor]) {
            dfs(neighbor, graph, visited, component);
        }
    }
}

std::vector<std::vector<int>> findConnectedComponents(const std::vector<std::vector<int>>& graph)
{
    std::vector<bool> visited(graph.size(), false);
    std::vector<std::vector<int>> components;

    for (int vertex = 0; vertex < graph.size(); ++vertex) {
        if (!visited[vertex]) {
            std::vector<int> component;
            dfs(vertex, graph, visited, component);
            components.push_back(component);
        }
    }

    return components;
}

std::vector<ClusterPtr> genClusters(const Eigen::MatrixXi& validation_matrix, const Eigen::MatrixXd& likelihood_matrix)
{
    // Validation matrix for all detections except null
    auto validation_matrix_true = validation_matrix(Eigen::all, Eigen::seqN(1, Eigen::last));

    // Initiate parameters
    int num_tracks = validation_matrix_true.rows();
    int num_detections = validation_matrix_true.cols();

    // Form clusters of tracks sharing measurements
    std::set<int> missed_tracks;
    std::vector<ClusterPtr> clusters;

    std::vector<std::vector<int>> v_list(num_detections);
    for (int detection = 0; detection < num_detections; detection++)
    {
        for (int track = 0; track < num_tracks; track++)
        {
            if (validation_matrix_true(track, detection) == 1)
            {
                v_list[detection].push_back(track);
            }
        }
    }

    std::vector<std::vector<int>> graph(num_tracks, std::vector<int>(num_tracks));
    for (auto& v : v_list)
    {
        if (v.size() == 0)
            continue;
        for (int i = 0; i < v.size()-1; i++)
        {
            int j = i + 1;
            graph[v[i]][v[j]] = 1;
            graph[v[j]][v[i]] = 1;
        }
    }

    std::vector<std::vector<int>> components = findConnectedComponents(graph);

    for (auto& tracks : components) {
        std::set<int> v_detections;
        for (auto& track : tracks)
        {
            for (int detection = 0; detection < num_detections; detection++)
            {
                if (validation_matrix_true(track, detection) == 1)
                {
                    v_detections.insert(detection + 1);
                }
            }
        }
        if (v_detections.size() == 0) {
            for (auto& track : tracks)
            {
                missed_tracks.insert(track);
            }
            continue;
        }
        v_detections.insert(0);
        std::vector<int> v_detections_vec(v_detections.begin(), v_detections.end());
        Eigen::MatrixXi c_validation_matrix = validation_matrix(tracks, v_detections_vec);
        Eigen::MatrixXd c_likelihood_matrix;
        if (likelihood_matrix.rows() == 0) {
            c_likelihood_matrix = Eigen::MatrixXd::Zero(0, 0);
        }
        else {
            c_likelihood_matrix = likelihood_matrix(tracks, v_detections_vec);
        }
        ClusterPtr c = std::make_shared<Cluster>(Cluster(tracks, v_detections_vec, c_validation_matrix, c_likelihood_matrix));
        clusters.push_back(c);
    }

    if (missed_tracks.size() > 0) {
        std::vector<int> missed_tracks_vec(missed_tracks.begin(), missed_tracks.end());
        ClusterPtr c = std::make_shared<Cluster>(Cluster(missed_tracks_vec));
        clusters.push_back(c);
    }

    return clusters;
}

} // namespace utils
} // namespace ehm
