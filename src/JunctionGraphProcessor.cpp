/*
  OSMScout ML
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

#include <osmscout/util/Geometry.h>
#include <osmscoutclient/json/json.hpp>

#include <JunctionGraphProcessor.h>

#include <fstream>

namespace osmscout {

using NodeIterator = std::list<RouteDescription::Node>::iterator;

struct GraphNode {
  Id id;
  GeoCoord location;
};

namespace GraphFeature{
constexpr std::string LANE_COUNT = "laneCount";
constexpr std::string ANGLE = "angle";
}

struct GraphEdge {
  Id fromNode;
  Id toNode;
  Distance length;
  std::unordered_map<std::string, double> features;
};

struct Graph {
  std::vector<GraphNode> nodes;
  std::vector<GraphEdge> edges;

  void Export(const std::filesystem::path &filePath) const {
    std::ofstream file(filePath);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file for writing: " + filePath.string());
    }

    nlohmann::json j;
    // Export nodes
    j["nodes"] = nlohmann::json::array();
    for (const auto& node : nodes) {
      j["nodes"].push_back({
        {"id", node.id},
        {"lat", node.location.GetLat()},
        {"lon", node.location.GetLon()}
      });
    }
    // Export edges
    j["edges"] = nlohmann::json::array();
    for (const auto& edge : edges) {
      auto edgeObj=nlohmann::json::object({
                                          {"from", edge.fromNode},
                                          {"to", edge.toNode},
                                          {"length", edge.length.AsMeter()}
                                        });
      for (const auto& feature : edge.features) {
        edgeObj[feature.first] = feature.second;
      }
      j["edges"].push_back(edgeObj);
    }
    file << j.dump(2) << std::endl;
    file.close();
  }
};

namespace {
Distance SegmentLength(const NodeIterator start,
                       const NodeIterator end) {
  Distance distance;
  auto it = start;
  while (it != end) {
    auto location = it->GetLocation();
    distance += GetSphericalDistance(location, (++it)->GetLocation());
  }
  return distance;
}

GraphNode CreateGraphNode(const RouteDescription::Node &node) {
  return GraphNode{
    node.GetLocation().GetId(),
    node.GetLocation()
  };
}
GraphEdge MakeEdge(const PostprocessorContext& context,
                   const NodeIterator prev,
                   const NodeIterator from,
                   const NodeIterator to) {
  GraphEdge edge{
    from->GetLocation().GetId(),
    to->GetLocation().GetId(),
    GetSphericalDistance(from->GetLocation(), to->GetLocation())
  };
  if (prev != from) {
    double inBearing=GetSphericalBearingFinal(prev->GetLocation(),from->GetLocation()).AsDegrees();
    double outBearing=GetSphericalBearingInitial(from->GetLocation(),to->GetLocation()).AsDegrees();

    double turnAngle=NormalizeRelativeAngle(outBearing - inBearing);
    edge.features[GraphFeature::ANGLE] = turnAngle;
  }
  if (auto laneDesc = from->GetDescription<RouteDescription::LaneDescription>();
      laneDesc && laneDesc->GetLaneCount() > 0) {
    edge.features[GraphFeature::LANE_COUNT] = laneDesc->GetLaneCount();
  }
  return edge;
}
} // anonymous namespace


JunctionGraphProcessor::JunctionGraphProcessor(const std::filesystem::path& exportDirectory):
  exportDirectory(exportDirectory)
{
  if (!std::filesystem::exists(exportDirectory)) {
    std::filesystem::create_directories(exportDirectory);
  }
  log.Debug() << "Junction graph export directory: " << exportDirectory;
}

bool JunctionGraphProcessor::Process(const PostprocessorContext& context,
                                     RouteDescription& description) {

  auto junctionStart = description.Nodes().begin();
  auto end = description.Nodes().end();
  for (auto nodeIt = description.Nodes().begin();
       nodeIt != end;
       ++nodeIt) {
    auto& node = *nodeIt;
    while (std::distance(junctionStart, nodeIt) > 1 &&
           SegmentLength(junctionStart, nodeIt) > Meters(50)) {
      assert(junctionStart != nodeIt);
      ++junctionStart;
    }

    if (node.HasDescription<RouteDescription::TurnDescription>() ||
        node.HasDescription<RouteDescription::MotorwayChangeDescription>() ||
        node.HasDescription<RouteDescription::MotorwayLeaveDescription>() ||
        node.HasDescription<RouteDescription::MotorwayJunctionDescription>()
        ) {

      // Fill the graph with nodes and edges
      Graph graph;
      Distance distanceAhead;
      bool ahead = false;
      auto prevNode = junctionStart;
      auto prevPrevNode = junctionStart;
      graph.nodes.push_back(CreateGraphNode(*prevNode));
      for (auto junctionNode = std::next(junctionStart);
           junctionNode != end && distanceAhead < Meters(50) && junctionNode->GetPathObject().Valid();
           ++junctionNode) {
        assert(prevNode!=junctionNode);

        // Create a graph node
        graph.nodes.push_back(CreateGraphNode(*junctionNode));

        // Create an edge to the junction start
        GraphEdge edge=MakeEdge(context, prevPrevNode, prevNode, junctionNode);
        if (ahead) {
          distanceAhead += edge.length;
        } else if (junctionNode == nodeIt) {
          ahead = true;
        }

        graph.edges.push_back(edge);

        prevPrevNode = prevNode;
        prevNode = junctionNode;
      }
      if (!graph.edges.empty()) {
        auto junctionFileName = std::to_string(node.GetPathObject().GetFileOffset()) + "_" + std::to_string(node.GetCurrentNodeIndex()) + ".json";
        log.Debug() << "Exporting junction graph for node "
                    << node.GetPathObject().GetFileOffset() << "/" << node.GetCurrentNodeIndex()
                    << " at " << node.GetLocation().GetDisplayText() << " to " << junctionFileName;
        graph.Export(exportDirectory / junctionFileName);
      }
      junctionStart = nodeIt;
    }
  }
  return true;
}

}
