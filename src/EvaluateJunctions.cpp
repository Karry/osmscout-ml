/*
  OSMScout ML - Junction Evaluation Tool
  Copyright (C) 2025  Lukáš Karas

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <iomanip>
#include <unordered_map>

#include <torch/torch.h>
#include <torch/script.h>

#include <JunctionGraphProcessor.h>

using namespace osmscout;

struct JunctionStats {
  std::string filename;
  size_t nodeCount = 0;
  size_t edgeCount = 0;
  size_t routeEdgeCount = 0;
  size_t matchCount = 0;
};

// Run prediction on a graph and return statistics
JunctionStats EvaluateJunction(const std::filesystem::path& jsonPath,
                               torch::jit::script::Module& model) {
  JunctionStats stats;
  stats.filename = jsonPath.filename().string();

  try {
    // Load the graph from JSON
    Graph graph;
    graph.Import(jsonPath);

    stats.nodeCount = graph.nodes.size();
    stats.edgeCount = graph.edges.size();

    if (graph.nodes.empty() || graph.edges.empty()) {
      return stats;
    }

    // Create node ID to index mapping
    std::unordered_map<Id, int> nodeIdToIndex;
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
      nodeIdToIndex[graph.nodes[i].id] = static_cast<int>(i);
    }

    // Prepare node features
    std::vector<std::vector<float>> nodeFeatures;
    for (const auto& graphNode : graph.nodes) {
      nodeFeatures.push_back({
        static_cast<float>(graphNode.normalizedLocation.GetLat()),
        static_cast<float>(graphNode.normalizedLocation.GetLon()),
        static_cast<float>(graphNode.incoming),
        static_cast<float>(graphNode.outgoing)
      });
    }

    // Prepare edge indices and features
    std::vector<std::vector<int64_t>> edgeIndices;
    std::vector<std::vector<float>> edgeFeatures;

    for (const auto& edge : graph.edges) {
      auto fromIt = nodeIdToIndex.find(edge.fromNode);
      auto toIt = nodeIdToIndex.find(edge.toNode);

      if (fromIt == nodeIdToIndex.end() || toIt == nodeIdToIndex.end()) {
        continue;
      }

      edgeIndices.push_back({static_cast<int64_t>(fromIt->second),
                            static_cast<int64_t>(toIt->second)});

      // Extract edge features in correct order
      std::vector<float> features;
      features.push_back(static_cast<float>(edge.length.AsMeter()));
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::LANE_COUNT) ?
                                           edge.features.at(GraphFeature::LANE_COUNT) : 0.0));
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::ANGLE) ?
                                           edge.features.at(GraphFeature::ANGLE) : 0.0));
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::ROUTE) ?
                                           edge.features.at(GraphFeature::ROUTE) : 0.0));
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::TYPE) ?
                                           edge.features.at(GraphFeature::TYPE) : -1.0));
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::USABLE) ?
                                           edge.features.at(GraphFeature::USABLE) : 0.0));
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::VIRTUAL) ?
                                           edge.features.at(GraphFeature::VIRTUAL) : 0.0));
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::RELATIVE_LANE_POSITION) ?
                                           edge.features.at(GraphFeature::RELATIVE_LANE_POSITION) : 0.0));
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::LANE_TURN) ?
                                           edge.features.at(GraphFeature::LANE_TURN) : -1.0));

      edgeFeatures.push_back(features);
    }

    if (edgeIndices.empty()) {
      return stats;
    }

    // Convert to PyTorch tensors
    const int nodeFeatureCount = 4;
    torch::Tensor nodeTensor = torch::zeros({static_cast<int64_t>(nodeFeatures.size()), nodeFeatureCount});
    for (size_t i = 0; i < nodeFeatures.size(); ++i) {
      for (int j = 0; j < nodeFeatureCount; ++j) {
        nodeTensor[i][j] = nodeFeatures[i][j];
      }
    }

    torch::Tensor edgeIndexTensor = torch::zeros({2, static_cast<int64_t>(edgeIndices.size())}, torch::kInt64);
    for (size_t i = 0; i < edgeIndices.size(); ++i) {
      edgeIndexTensor[0][i] = edgeIndices[i][0];
      edgeIndexTensor[1][i] = edgeIndices[i][1];
    }

    torch::Tensor edgeAttrTensor = torch::zeros({static_cast<int64_t>(edgeFeatures.size()), GraphFeature::EdgeFeatureCount});
    for (size_t i = 0; i < edgeFeatures.size(); ++i) {
      for (size_t j = 0; j < edgeFeatures[i].size() && j < GraphFeature::EdgeFeatureCount; ++j) {
        edgeAttrTensor[i][j] = edgeFeatures[i][j];
      }
    }

    // Run model inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(nodeTensor);
    inputs.push_back(edgeIndexTensor);
    inputs.push_back(edgeAttrTensor);

    torch::jit::IValue output = model.forward(inputs);
    torch::Tensor suggestedPred = torch::sigmoid(output.toTensor());

    // Compare predictions with heuristics for ROUTE edges
    for (size_t i = 0; i < graph.edges.size() && i < edgeIndices.size(); ++i) {
      const auto& edge = graph.edges[i];

      // Check if edge is on route
      bool isRoute = edge.features.contains(GraphFeature::ROUTE) &&
                     edge.features.at(GraphFeature::ROUTE) > 0.5;

      if (isRoute) {
        stats.routeEdgeCount++;

        // Get heuristic suggestion
        bool heuristicSuggested = edge.features.contains(GraphFeature::SUGGESTED) &&
                                  edge.features.at(GraphFeature::SUGGESTED) > 0.5;

        // Get model prediction
        float predValue = suggestedPred[i].item<float>();
        bool modelSuggested = predValue > 0.5;

        // Count matches
        if (heuristicSuggested == modelSuggested) {
          stats.matchCount++;
        }
      }
    }

  } catch (const std::exception& e) {
    std::cerr << "Error processing " << stats.filename << ": " << e.what() << std::endl;
  }

  return stats;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <junction_json_directory> <pytorch_model_path>" << std::endl;
    std::cerr << "Example: " << argv[0] << " tmp-junctions/ model.pt" << std::endl;
    return 1;
  }

  std::filesystem::path jsonDir = argv[1];
  std::filesystem::path modelPath = argv[2];

  // Validate inputs
  if (!std::filesystem::exists(jsonDir) || !std::filesystem::is_directory(jsonDir)) {
    std::cerr << "Error: Directory does not exist: " << jsonDir << std::endl;
    return 1;
  }

  if (!std::filesystem::exists(modelPath)) {
    std::cerr << "Error: Model file does not exist: " << modelPath << std::endl;
    return 1;
  }

  // Load PyTorch model
  std::cout << "Loading PyTorch model from: " << modelPath << std::endl;
  torch::jit::script::Module model;
  try {
    model = torch::jit::load(modelPath.string());
    model.eval();
    std::cout << "Model loaded successfully!" << std::endl;
  } catch (const c10::Error& e) {
    std::cerr << "Error loading model: " << e.what() << std::endl;
    return 1;
  }

  // Collect all JSON files
  std::vector<std::filesystem::path> jsonFiles;
  for (const auto& entry : std::filesystem::directory_iterator(jsonDir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".json") {
      jsonFiles.push_back(entry.path());
    }
  }

  if (jsonFiles.empty()) {
    std::cerr << "No JSON files found in directory: " << jsonDir << std::endl;
    return 1;
  }

  std::cout << "Found " << jsonFiles.size() << " JSON files to process" << std::endl;
  std::cout << std::endl;

  // Sort files by name for consistent output
  std::sort(jsonFiles.begin(), jsonFiles.end());

  // Print table header
  std::cout << std::left
            << std::setw(30) << "Filename"
            << std::setw(10) << "Nodes"
            << std::setw(10) << "Edges"
            << std::setw(15) << "Route Edges"
            << std::setw(15) << "Matches"
            << std::setw(15) << "Accuracy"
            << std::endl;
  std::cout << std::string(95, '-') << std::endl;

  // Process each JSON file
  std::vector<JunctionStats> allStats;
  for (const auto& jsonFile : jsonFiles) {
    JunctionStats stats = EvaluateJunction(jsonFile, model);
    allStats.push_back(stats);

    // Calculate accuracy
    double accuracy = 0.0;
    if (stats.routeEdgeCount > 0) {
      accuracy = (100.0 * stats.matchCount) / stats.routeEdgeCount;
    }

    // Print row
    std::cout << std::left
              << std::setw(30) << stats.filename
              << std::setw(10) << stats.nodeCount
              << std::setw(10) << stats.edgeCount
              << std::setw(15) << stats.routeEdgeCount
              << std::setw(15) << stats.matchCount;

    if (stats.routeEdgeCount > 0) {
      std::cout << std::fixed << std::setprecision(1)
                << std::setw(15) << accuracy << "%";
    } else {
      std::cout << std::setw(15) << "N/A";
    }
    std::cout << std::endl;
  }

  // Print summary statistics
  std::cout << std::string(95, '-') << std::endl;

  size_t totalNodes = 0;
  size_t totalEdges = 0;
  size_t totalRouteEdges = 0;
  size_t totalMatches = 0;

  for (const auto& stats : allStats) {
    totalNodes += stats.nodeCount;
    totalEdges += stats.edgeCount;
    totalRouteEdges += stats.routeEdgeCount;
    totalMatches += stats.matchCount;
  }

  double overallAccuracy = 0.0;
  if (totalRouteEdges > 0) {
    overallAccuracy = (100.0 * totalMatches) / totalRouteEdges;
  }

  std::cout << std::left
            << std::setw(30) << "TOTAL"
            << std::setw(10) << totalNodes
            << std::setw(10) << totalEdges
            << std::setw(15) << totalRouteEdges
            << std::setw(15) << totalMatches
            << std::fixed << std::setprecision(1)
            << std::setw(15) << overallAccuracy << "%"
            << std::endl;

  std::cout << "\nProcessed " << allStats.size() << " junction files" << std::endl;
  std::cout << "Overall accuracy: " << std::fixed << std::setprecision(2)
            << overallAccuracy << "%" << std::endl;

  return 0;
}

