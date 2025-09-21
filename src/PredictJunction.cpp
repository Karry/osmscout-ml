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
        static_cast<float>(graphNode.location.GetLat()),
        static_cast<float>(graphNode.location.GetLon())
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
      features.push_back(static_cast<float>(edge.length.AsMeter())); // length
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::LANE_COUNT) ?
                                           edge.features.at(GraphFeature::LANE_COUNT) : 0.0)); // laneCount
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::ANGLE) ?
                                           edge.features.at(GraphFeature::ANGLE) : 0.0)); // angle
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::ONEWAY) ?
                                           edge.features.at(GraphFeature::ONEWAY) : 0.0)); // oneway
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::ROUTE) ?
                                           edge.features.at(GraphFeature::ROUTE) : 0.0)); // route
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::TYPE) ?
                                           edge.features.at(GraphFeature::TYPE) : -1.0)); // type
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::USABLE) ?
                                           edge.features.at(GraphFeature::USABLE) : 0.0)); // usable
      features.push_back(static_cast<float>(edge.features.contains(GraphFeature::VIRTUAL) ?
                                           edge.features.at(GraphFeature::VIRTUAL) : 0.0)); // virtual

      // Lane turn features (up to 10 lanes, as expected by the model)
      for (int i = 0; i < 10; ++i) {
        std::string laneTurnKey = "laneTurn" + std::to_string(i);
        features.push_back(static_cast<float>(edge.features.contains(laneTurnKey) ?
                                             edge.features.at(laneTurnKey) : -1.0));
      }

      edgeFeatures.push_back(features);
    }

    if (edgeIndices.empty()) {
      std::cout << "No valid edges found for prediction" << std::endl;
      return;
    }

    // Convert to PyTorch tensors
    // Node features tensor [num_nodes, 2]
    torch::Tensor nodeTensor = torch::zeros({static_cast<int64_t>(nodeFeatures.size()), 2});
    for (size_t i = 0; i < nodeFeatures.size(); ++i) {
      nodeTensor[i][0] = nodeFeatures[i][0]; // lat
      nodeTensor[i][1] = nodeFeatures[i][1]; // lon
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

    // Extract predictions from output tuple (not dictionary)
    auto outputTuple = output.toTuple();
    torch::Tensor suggestedFromPred = outputTuple->elements()[0].toTensor();
    torch::Tensor suggestedToPred = outputTuple->elements()[1].toTensor();
    torch::Tensor suggestedTurnPred = outputTuple->elements()[2].toTensor();

    // Apply sigmoid to get probabilities for binary predictions
    suggestedFromPred = torch::sigmoid(suggestedFromPred);
    suggestedToPred = torch::sigmoid(suggestedToPred);
    // suggestedTurn might be a regression output, so we don't apply sigmoid

    // Print predictions alongside heuristic suggestions for each edge
    std::cout << "\n=== Edge Predictions vs Heuristics ===" << std::endl;
    for (size_t i = 0; i < graph.edges.size() && i < edgeIndices.size(); ++i) {
      const auto& edge = graph.edges[i];
      if (edge.features.contains(GraphFeature::VIRTUAL) &&
          edge.features.at(GraphFeature::VIRTUAL) == 1.0) {
        continue; // Skip virtual edges
      }

      std::cout << "\nEdge " << i << " (from " << edge.fromNode << " to " << edge.toNode << "):" << std::endl;

      // Print model predictions
      float predFrom = suggestedFromPred[i].item<float>();
      float predTo = suggestedToPred[i].item<float>();
      float predTurn = suggestedTurnPred[i].item<float>();

      std::cout << "  Model Predictions:" << std::endl;
      std::cout << "    suggestedFrom: " << std::fixed << std::setprecision(3) << predFrom << std::endl;
      std::cout << "    suggestedTo:   " << std::fixed << std::setprecision(3) << predTo << std::endl;
      std::cout << "    suggestedTurn: " << std::fixed << std::setprecision(3) << predTurn << " (" <<  LaneTurnString(LaneTurn(uint8_t(predTurn))) << ")" << std::endl;

      // Print heuristic suggestions (if available)
      std::cout << "  Heuristic Values:" << std::endl;
      if (edge.features.contains(GraphFeature::SUGGESTED_FROM)) {
        std::cout << "    suggestedFrom: " << edge.features.at(GraphFeature::SUGGESTED_FROM) << std::endl;
      } else {
        std::cout << "    suggestedFrom: (not available)" << std::endl;
      }

      if (edge.features.contains(GraphFeature::SUGGESTED_TO)) {
        std::cout << "    suggestedTo:   " << edge.features.at(GraphFeature::SUGGESTED_TO) << std::endl;
      } else {
        std::cout << "    suggestedTo:   (not available)" << std::endl;
      }

      if (edge.features.contains(GraphFeature::SUGGESTED_TURN)) {
        float heurTurn = edge.features.at(GraphFeature::SUGGESTED_TURN);
        std::cout << "    suggestedTurn: " << heurTurn << " (" << LaneTurnString(LaneTurn(uint8_t(heurTurn))) << ")"  << std::endl;
      } else {
        std::cout << "    suggestedTurn: (not available)" << std::endl;
      }

      // Print other relevant features for context
      // [length, laneCount, angle, oneway, route, type, laneTurn0-9]
      std::cout << "  Context:" << std::endl;
      std::cout << "    length: " << edge.length.AsMeter() << "m" << std::endl;
      if (edge.features.contains(GraphFeature::LANE_COUNT)) {
        std::cout << "    laneCount: " << edge.features.at(GraphFeature::LANE_COUNT) << std::endl;
      }
      if (edge.features.contains(GraphFeature::ANGLE)) {
        std::cout << "    angle: " << edge.features.at(GraphFeature::ANGLE) << std::endl;
      }
      if (edge.features.contains(GraphFeature::ONEWAY)) {
        std::cout << "    oneway: " << (edge.features.at(GraphFeature::ONEWAY) > 0 ? "yes" : "no") << std::endl;
      }
      if (edge.features.contains(GraphFeature::ROUTE)) {
        std::cout << "    route: " << (edge.features.at(GraphFeature::ROUTE) > 0 ? "yes" : "no") << std::endl;
      }
      if (edge.features.contains(GraphFeature::TYPE)) {
        std::cout << "    type: " << edge.features.at(GraphFeature::TYPE) << " (" << GraphFeature::WayTypeName(edge.features.at(GraphFeature::TYPE)) << ")" << std::endl;
      }
      for (int j = 0; j < 10; ++j) {
        std::string laneTurnKey = "laneTurn" + std::to_string(j);
        if (edge.features.contains(laneTurnKey)) {
          float laneTurnVal = edge.features.at(laneTurnKey);
          std::cout << "    " << laneTurnKey << ": " << laneTurnVal << " (" << LaneTurnString(LaneTurn(uint8_t(laneTurnVal))) << ")" << std::endl;
        }
      }
      if (edge.features.contains(GraphFeature::USABLE)) {
        std::cout << "    usable: " << (edge.features.at(GraphFeature::USABLE) > 0 ? "yes" : "no") << std::endl;
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
