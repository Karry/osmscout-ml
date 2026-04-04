/*
  OSMScout ML
  Copyright (C) 2025  Lukáš Karas
  Copyright (C) 2009  Tim Teulings

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
#include <chrono>
#include <cstring>
#include <iostream>
#include <list>
#include <fstream>
#include <optional>
#include <filesystem>
#include <iomanip>
#include <unordered_map>

#include <osmscout/db/Database.h>

#include <osmscout/routing/SimpleRoutingService.h>
#include <osmscout/routing/RoutePostprocessor.h>
#include <osmscout/routing/DBFileOffset.h>
#include <osmscout/routing/RouteDescriptionPostprocessor.h>

#include <osmscout/cli/CmdLineParsing.h>
#include <osmscout/util/Bearing.h>
#include <osmscout/util/Geometry.h>

#include <JunctionGraphProcessor.h>
#include <ConsoleRoutingProgress.h>
#include <RoutingUtils.h>

// PyTorch++ includes
#include <torch/torch.h>
#include <torch/script.h>

struct Arguments
{
  bool                              help=false;
  std::string                       router=osmscout::RoutingService::DEFAULT_FILENAME_BASE;
  osmscout::Vehicle                 vehicle=osmscout::Vehicle::vehicleCar;
  std::string                       gpx;
  std::string                       databaseDirectory;
  std::string                       modelPath;  // PyTorch model path
  osmscout::GeoCoord                start;
  std::vector<osmscout::GeoCoord>   via;
  osmscout::GeoCoord                target;
  std::optional<osmscout::Bearing>  initialBearing;
  bool                              debug=false;
  bool                              dataDebug=false;
  bool                              routeDebug=false;
  std::string                       routeJson;

  osmscout::Distance                penaltySameType=osmscout::Meters(40);
  osmscout::Distance                penaltyDifferentType=osmscout::Meters(250);
  osmscout::HourDuration            maxPenalty=std::chrono::seconds(10);
};

namespace osmscout {
class JunctionGraphPredictProcessor: public JunctionGraphProcessor {
private:
  torch::jit::script::Module model;
public:
  explicit JunctionGraphPredictProcessor(torch::jit::script::Module &&model);
  ~JunctionGraphPredictProcessor() override = default;

  JunctionGraphPredictProcessor(const JunctionGraphPredictProcessor&) = delete;
  JunctionGraphPredictProcessor& operator=(const JunctionGraphPredictProcessor&) = delete;

  JunctionGraphPredictProcessor(JunctionGraphPredictProcessor&&) = delete;
  JunctionGraphPredictProcessor& operator=(JunctionGraphPredictProcessor&&) = delete;

  /** Process the junction graph using the ML model
   *
   * @param graph - normalized graph
   * @param node
   */
  void ProcessJunctionGraph(const Graph &graph,
                            const RouteDescription::Node &node) override;
};

using JunctionGraphPredictProcessorRef = std::shared_ptr<JunctionGraphPredictProcessor>;

JunctionGraphPredictProcessor::JunctionGraphPredictProcessor(torch::jit::script::Module &&model):
  model(std::move(model))
{
}

void JunctionGraphPredictProcessor::ProcessJunctionGraph(const Graph &graph,
                                                         const RouteDescription::Node &node)
{
  // Check if we have any nodes and edges in the graph
  if (graph.nodes.empty() || graph.edges.empty()) {
    std::cout << "Empty graph - skipping prediction" << std::endl;
    return;
  }

  std::cout << "=== Junction Graph Prediction ===" << std::endl;
  std::cout << "Nodes: " << graph.nodes.size() << ", Edges: " << graph.edges.size() << std::endl;

  try {
    // Create node ID to index mapping for PyTorch tensor indexing
    std::unordered_map<Id, int> nodeIdToIndex;
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
      nodeIdToIndex[graph.nodes[i].id] = static_cast<int>(i);
    }

    // Prepare node features (lat, lon coordinates)
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
      // Get node indices for this edge
      auto fromIt = nodeIdToIndex.find(edge.fromNode);
      auto toIt = nodeIdToIndex.find(edge.toNode);

      // check that Edge references known node
      assert(fromIt != nodeIdToIndex.end() && toIt != nodeIdToIndex.end());

      // Add edge indices (from_node, to_node)
      edgeIndices.push_back({static_cast<int64_t>(fromIt->second), static_cast<int64_t>(toIt->second)});

      // Extract edge features in the same order as the Python model expects:
      // [length, laneCount, angle, oneway, route, type, usable, laneTurn0-9]
      std::vector<float> features;

      // Basic features
      // New lane-level features (9 total)
      features.push_back(static_cast<float>(edge.length.AsMeter())); // 1. length
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::LANE_COUNT) ?
                                           edge.features.at(GraphFeature::LANE_COUNT) : 0.0)); // 2. laneCount
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::ANGLE) ?
                                           edge.features.at(GraphFeature::ANGLE) : 0.0)); // 3. angle
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::ROUTE) ?
                                           edge.features.at(GraphFeature::ROUTE) : 0.0)); // 4. route
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::TYPE) ?
                                           edge.features.at(GraphFeature::TYPE) : -1.0)); // 5. type
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::USABLE) ?
                                           edge.features.at(GraphFeature::USABLE) : 0.0)); // 6. usable
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::VIRTUAL) ?
                                           edge.features.at(GraphFeature::VIRTUAL) : 0.0)); // 7. virtual
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::RELATIVE_LANE_POSITION) ?
                                           edge.features.at(GraphFeature::RELATIVE_LANE_POSITION) : 0.0)); // 8. relativeLanePosition
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::LANE_TURN) ?
                                           edge.features.at(GraphFeature::LANE_TURN) : -1.0)); // 9. laneTurn

      edgeFeatures.push_back(features);
    }

    if (edgeIndices.empty()) {
      std::cout << "No valid edges found for prediction" << std::endl;
      return;
    }

    // Convert to PyTorch tensors
    // Node features tensor [num_nodes, 4] (lat, lon, incoming, outgoing)
    const int nodeFeatureCount = 4;
    torch::Tensor nodeTensor = torch::zeros({static_cast<int64_t>(nodeFeatures.size()), nodeFeatureCount});
    for (size_t i = 0; i < nodeFeatures.size(); ++i) {
      for (int j = 0; j < nodeFeatureCount; ++j) {
        nodeTensor[i][j] = nodeFeatures[i][j];
      }
    }

    // Edge index tensor [2, num_edges] - PyTorch Geometric format
    torch::Tensor edgeIndexTensor = torch::zeros({2, static_cast<int64_t>(edgeIndices.size())}, torch::kInt64);
    for (size_t i = 0; i < edgeIndices.size(); ++i) {
      edgeIndexTensor[0][i] = edgeIndices[i][0]; // from_node
      edgeIndexTensor[1][i] = edgeIndices[i][1]; // to_node
    }

    // Edge attributes tensor [num_edges, GraphFeature::EdgeFeatureCount]
    torch::Tensor edgeAttrTensor = torch::zeros({static_cast<int64_t>(edgeFeatures.size()), GraphFeature::EdgeFeatureCount});
    for (size_t i = 0; i < edgeFeatures.size(); ++i) {
      for (size_t j = 0; j < edgeFeatures[i].size() && j < GraphFeature::EdgeFeatureCount; ++j) {
        edgeAttrTensor[i][j] = edgeFeatures[i][j];
      }
    }

    // Create input arguments for the model (individual tensors, not dictionary)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(nodeTensor);          // node_features
    inputs.push_back(edgeIndexTensor);     // edge_index
    inputs.push_back(edgeAttrTensor);      // edge_features

    // Run model inference
    std::cout << "Running model inference..." << std::endl;
    torch::jit::IValue output = model.forward(inputs);

    // Extract single prediction tensor (binary: is this lane suggested?)
    torch::Tensor suggestedPred = output.toTensor();

    // Apply sigmoid to get probabilities for binary classification
    suggestedPred = torch::sigmoid(suggestedPred);

    // Print lane-level predictions
    std::cout << "\n=== Lane Predictions (per-lane edges) ===" << std::endl;

    // Group edges by highway (fromNode -> toNode pair)
    std::map<std::pair<Id, Id>, std::vector<size_t>> edgesByHighway;
    for (size_t i = 0; i < graph.edges.size() && i < edgeIndices.size(); ++i) {
      const auto& edge = graph.edges[i];
      if (edge.features.contains(GraphFeature::VIRTUAL) &&
          edge.features.at(GraphFeature::VIRTUAL) == 1.0) {
        continue; // Skip virtual edges
      }
      edgesByHighway[{edge.fromNode, edge.toNode}].push_back(i);
    }

    // Print grouped by highway
    for (const auto& [highway, laneIndices] : edgesByHighway) {
      std::cout << "\n--- Highway from Node " << highway.first << " to Node " << highway.second << " ---" << std::endl;

      // Print highway-level info from first lane
      const auto& firstEdge = graph.edges[laneIndices[0]];
      std::cout << "  Highway info:" << std::endl;
      std::cout << "    length: " << firstEdge.length.AsMeter() << "m" << std::endl;
      if (firstEdge.features.contains(GraphFeature::LANE_COUNT)) {
        std::cout << "    total lanes: " << static_cast<int>(firstEdge.features.at(GraphFeature::LANE_COUNT)) << std::endl;
      }
      if (firstEdge.features.contains(GraphFeature::ANGLE)) {
        std::cout << "    angle: " << std::fixed << std::setprecision(1) << firstEdge.features.at(GraphFeature::ANGLE) << "°" << std::endl;
      }
      if (firstEdge.features.contains(GraphFeature::TYPE)) {
        std::cout << "    type: " << GraphFeature::WayTypeName(firstEdge.features.at(GraphFeature::TYPE)) << std::endl;
      }
      if (firstEdge.features.contains(GraphFeature::ROUTE)) {
        std::cout << "    on route: " << (firstEdge.features.at(GraphFeature::ROUTE) > 0 ? "yes" : "no") << std::endl;
      }

      std::cout << "\n  Lanes:" << std::endl;

      // Print each lane
      for (size_t idx : laneIndices) {
        const auto& edge = graph.edges[idx];

        // Get lane-specific features
        double lanePos = edge.features.contains(GraphFeature::RELATIVE_LANE_POSITION)
                         ? edge.features.at(GraphFeature::RELATIVE_LANE_POSITION) : -1.0;
        double laneTurn = edge.features.contains(GraphFeature::LANE_TURN)
                          ? edge.features.at(GraphFeature::LANE_TURN) : -1.0;
        double heuristicSuggested = edge.features.contains(GraphFeature::SUGGESTED)
                                    ? edge.features.at(GraphFeature::SUGGESTED) : -1.0;

        // Get model prediction
        float predSuggested = suggestedPred[idx].item<float>();

        // Format lane position
        std::string posStr;
        if (lanePos < 0.33) posStr = "left ";
        else if (lanePos > 0.67) posStr = "right";
        else posStr = "mid  ";

        // Print lane info
        std::cout << "    Lane (pos=" << std::fixed << std::setprecision(2) << lanePos << ", " << posStr << "): ";
        std::cout << "turn=" << LaneTurnString(LaneTurn(uint8_t(laneTurn)));
        std::cout << ", predicted=" << std::fixed << std::setprecision(3) << predSuggested;
        std::cout << (predSuggested > 0.5 ? " ✓" : "");

        if (heuristicSuggested >= 0) {
          std::cout << ", heuristic=" << (heuristicSuggested > 0.5 ? "YES" : "NO");
        }
        std::cout << std::endl;
      }
    }

    std::cout << "\n=== End Junction Prediction ===" << std::endl;

  } catch (const std::exception& e) {
    std::cout << "Error during prediction: " << e.what() << std::endl;
  }
}

}

int main(int argc, char* argv[]) {
  using namespace osmscout;
  using namespace std::string_literals;
  using namespace std::chrono;

  osmscout::CmdLineParser   argParser("Junction Prediction",
                                      argc,argv);
  std::vector<std::string>  helpArgs{"h","help"};
  Arguments                 args;

  argParser.AddOption(osmscout::CmdLineFlag([&args](const bool& value) {
                        args.help=value;
                      }),
                      helpArgs,
                      "Return argument help",
                      true);

  argParser.AddOption(osmscout::CmdLineStringOption([&args](const std::string& value) {
                        args.gpx=value;
                      }),
                      "gpx",
                      "Dump resulting route as GPX to file",
                      false);

  argParser.AddOption(osmscout::CmdLineStringOption([&args](const std::string& value) {
                        args.routeJson=value;
                      }),
                      "routeJson",
                      "Dump resulting route as JSON to file",
                      false);

  argParser.AddOption(osmscout::CmdLineStringOption([&args](const std::string& value) {
                        args.modelPath=value;
                      }),
                      "model",
                      "Path to PyTorch model file",
                      false);

  argParser.AddOption(osmscout::CmdLineFlag([&args](const bool& value) {
                        args.debug=value;
                      }),
                      "debug",
                      "Enable debug output",
                      false);

  argParser.AddOption(osmscout::CmdLineFlag([&args](const bool& value) {
                        args.dataDebug=value;
                      }),
                      "dataDebug",
                      "Dump data nodes to std::cout",
                      false);

  argParser.AddOption(osmscout::CmdLineFlag([&args](const bool& value) {
                        args.routeDebug=value;
                      }),
                      "routeDebug",
                      "Dump route description data to std::cout",
                      false);

  argParser.AddOption(osmscout::CmdLineAlternativeFlag([&args](const std::string& value) {
                        if (value=="foot") {
                          args.vehicle=osmscout::Vehicle::vehicleFoot;
                        }
                        else if (value=="bicycle") {
                          args.vehicle=osmscout::Vehicle::vehicleBicycle;
                        }
                        else if (value=="car") {
                          args.vehicle=osmscout::Vehicle::vehicleCar;
                        }
                      }),
                      {"foot","bicycle","car"},
                      "Vehicle type to use for routing");

  argParser.AddOption(osmscout::CmdLineStringOption([&args](const std::string& value) {
                        args.router=value;
                      }),
                      "router",
                      "Router filename base");

  argParser.AddOption(osmscout::CmdLineUIntOption([&args](unsigned int value) {
                        args.penaltySameType=osmscout::Meters(value);
                      }),
                      "penalty-same",
                      "Junction penalty for same types, distance [m]. Default "s + std::to_string((int)args.penaltySameType.AsMeter()));

  argParser.AddOption(osmscout::CmdLineUIntOption([&args](unsigned int value) {
                        args.penaltyDifferentType=osmscout::Meters(value);
                      }),
                      "penalty-diff",
                      "Junction penalty for different types, distance [m]. Default "s + std::to_string((int)args.penaltyDifferentType.AsMeter()));

  argParser.AddOption(osmscout::CmdLineUIntOption([&args](unsigned int value) {
                        args.maxPenalty=seconds(value);
                      }),
                      "penalty-max",
                      "Maximum junction penalty, time [s]. Default "s + std::to_string(duration_cast<seconds>(args.maxPenalty).count()));

  argParser.AddOption(osmscout::CmdLineDoubleOption([&args](double value) {
                        args.initialBearing=osmscout::Bearing::Degrees(value);
                      }),
                      "initial-bearing",
                      "Initial vehicle bearing (degrees, North is 0, East is 90...).");

  argParser.AddOption(osmscout::CmdLineGeoCoordOption([&args](const osmscout::GeoCoord& value) {
                        args.via.push_back(value);
                      }),
                      "via",
                      "add a via location coordinate");

  argParser.AddPositional(osmscout::CmdLineStringOption([&args](const std::string& value) {
                            args.databaseDirectory=value;
                          }),
                          "DATABASE",
                          "Directory of the first db to use");

  argParser.AddPositional(osmscout::CmdLineGeoCoordOption([&args](const osmscout::GeoCoord& value) {
                            args.start=value;
                          }),
                          "START",
                          "start coordinate");

  argParser.AddPositional(osmscout::CmdLineGeoCoordOption([&args](const osmscout::GeoCoord& value) {
                            args.target=value;
                          }),
                          "TARGET",
                          "target coordinate");

  osmscout::CmdLineParseResult cmdLineParseResult=argParser.Parse();

  if (cmdLineParseResult.HasError()) {
    std::cerr << "ERROR: " << cmdLineParseResult.GetErrorDescription() << std::endl;
    std::cout << argParser.GetHelp() << std::endl;
    return 1;
  }

  if (args.help) {
    std::cout << argParser.GetHelp() << std::endl;
    return 0;
  }

  if (args.modelPath.empty()) {
    std::cerr << "Error: Model is required!" << std::endl;
    return 1;
  }

  osmscout::log.Debug(args.debug);
  osmscout::log.Info(true);
  osmscout::log.Warn(true);
  osmscout::log.Error(true);

  osmscout::log.Info() << "Model path: " << args.modelPath;

  torch::jit::script::Module model;
  try {
    model = torch::jit::load(args.modelPath);
  } catch (const std::exception& e) {
    osmscout::log.Error() << "Error loading model: " << e.what();
    return 1;
  }

  osmscout::log.Info() << "Database: " << args.databaseDirectory;
  osmscout::log.Info() << "Start: " << args.start.GetDisplayText();
  osmscout::log.Info() << "Target: " << args.target.GetDisplayText();

  std::list<osmscout::RoutePostprocessor::PostprocessorRef> postprocessors{
    std::make_shared<osmscout::RoutePostprocessor::DistanceAndTimePostprocessor>(),
    std::make_shared<osmscout::RoutePostprocessor::StartPostprocessor>("Start"),
    std::make_shared<osmscout::RoutePostprocessor::TargetPostprocessor>("Target"),
    std::make_shared<osmscout::RoutePostprocessor::WayNamePostprocessor>(),
    std::make_shared<osmscout::RoutePostprocessor::WayTypePostprocessor>(),
    std::make_shared<osmscout::RoutePostprocessor::CrossingWaysPostprocessor>(),
    std::make_shared<osmscout::RoutePostprocessor::DirectionPostprocessor>(),
    std::make_shared<osmscout::RoutePostprocessor::LanesPostprocessor>(),
    std::make_shared<osmscout::RoutePostprocessor::SuggestedLanesPostprocessor>(),
    std::make_shared<osmscout::RoutePostprocessor::MotorwayJunctionPostprocessor>(),
    std::make_shared<osmscout::RoutePostprocessor::DestinationPostprocessor>(),
    std::make_shared<osmscout::RoutePostprocessor::MaxSpeedPostprocessor>(),
    std::make_shared<osmscout::RoutePostprocessor::InstructionPostprocessor>(),
    std::make_shared<osmscout::RoutePostprocessor::POIsPostprocessor>(),
    std::make_shared<osmscout::JunctionGraphPredictProcessor>(std::move(model)),
  };

  return ComputeRoute(args.databaseDirectory,
                      postprocessors,
                      args.start,
                      args.target,
                      args.via,
                      args.penaltySameType,
                      args.penaltyDifferentType,
                      args.maxPenalty,
                      args.router,
                      args.vehicle,
                      args.initialBearing,
                      args.dataDebug);
}
