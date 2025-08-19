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
constexpr std::string ONEWAY = "oneway";
constexpr std::string SUGGESTED_FROM = "suggestedFrom";
constexpr std::string SUGGESTED_TO = "suggestedTo";
constexpr std::string SUGGESTED_TURN = "suggestedTurn";
constexpr std::string ROUTE = "route";
constexpr std::string TYPE = "type";
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

  std::set<Id> nodeIdSet;

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

  bool InsertNode(GraphNode node) {
    if (nodeIdSet.find(node.id) != nodeIdSet.end()) {
      return false;
    }
    nodes.push_back(node);
    nodeIdSet.insert(node.id);
    return true;
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

GraphNode CreateGraphNode(const PostprocessorContext &context, const RouteDescription::Node &node) {
  return GraphNode{
    context.GetNodeId(node),
    node.GetLocation()
  };
}

int WayTypeId(const std::string& typeName) {
  if (typeName == "highway_motorway")
    return 0;
  if (typeName == "highway_motorway_link")
    return 1;
  if (typeName == "highway_motorway_trunk")
    return 2;
  if (typeName == "highway_tertiary")
    return 3;
  if (typeName == "highway_trunk_link")
    return 4;

  log.Warn() << "Unknown way type: " << typeName;
  return -1; // Unknown type
}

GraphEdge MakeEdge(const PostprocessorContext& context,
                   const NodeIterator prev,
                   const NodeIterator from,
                   const NodeIterator to) {
  GraphEdge edge{
    context.GetNodeId(*from),
    context.GetNodeId(*to),
    GetSphericalDistance(from->GetLocation(), to->GetLocation())
  };
  edge.features[GraphFeature::ROUTE] = 1.0; // Mark this edge as part of the route
  if (from->GetPathObject().IsWay()) {
    edge.features[GraphFeature::TYPE] = WayTypeId(context.GetWay(from->GetDBFileOffset())->GetType()->GetName());
  }
  if (prev != from) {
    double inBearing=GetSphericalBearingFinal(prev->GetLocation(),from->GetLocation()).AsDegrees();
    double outBearing=GetSphericalBearingInitial(from->GetLocation(),to->GetLocation()).AsDegrees();

    double turnAngle=NormalizeRelativeAngle(outBearing - inBearing);
    edge.features[GraphFeature::ANGLE] = turnAngle;
  }
  if (auto laneDesc = from->GetDescription<RouteDescription::LaneDescription>();
      laneDesc && laneDesc->GetLaneCount() > 0) {
    edge.features[GraphFeature::LANE_COUNT] = laneDesc->GetLaneCount();
    edge.features[GraphFeature::ONEWAY] = laneDesc->IsOneway() ? 1.0 : 0.0;
    for (int i=0; i<laneDesc->GetLaneCount(); ++i) {
      if (i < laneDesc->GetLaneTurns().size()) {
        edge.features["laneTurn"+std::to_string(i)] = static_cast<double>(laneDesc->GetLaneTurns()[i]);
      } else {
        edge.features["laneTurn"+std::to_string(i)] = static_cast<double>(LaneTurn::Unknown);
      }
    }
    if (auto suggestedLanes = from->GetDescription<RouteDescription::SuggestedLaneDescription>();
        suggestedLanes) {
      edge.features[GraphFeature::SUGGESTED_FROM] = static_cast<double>(suggestedLanes->GetFrom());
      edge.features[GraphFeature::SUGGESTED_TO] = static_cast<double>(suggestedLanes->GetTo());
      edge.features[GraphFeature::SUGGESTED_TURN] = static_cast<double>(suggestedLanes->GetTurn());
    }
  }
  return edge;
}

void TraverseWay(const PostprocessorContext &context,
                 osmscout::Graph &graph,
                 DatabaseId dbId,
                 const NodeIterator prev,
                 const WayRef &way,
                 size_t id,
                 int direction) {
  assert(direction == -1 || direction == 1);
  assert(way);
  assert(id < way->nodes.size());
  assert(id < std::numeric_limits<int64_t>::max());

  auto laneDesc = context.GetLaneReader(dbId).GetValue(way->GetFeatureValueBuffer());
  auto accessDesc = context.GetAccessReader(dbId).GetValue(way->GetFeatureValueBuffer());

  Distance distance;
  for (auto i = int64_t(id);
       i < way->nodes.size() && i >= 0 && (i+direction) < way->nodes.size() && (i+direction) >= 0;
       i += direction) {

    const auto &from = way->nodes[i];
    const auto &to = way->nodes[i+direction];
    graph.InsertNode(GraphNode{from.GetId(), from.GetCoord()});
    graph.InsertNode(GraphNode{to.GetId(), to.GetCoord()});

    auto edge = GraphEdge{
      from.GetId(),
      to.GetId(),
      GetSphericalDistance(from.GetCoord(), to.GetCoord())
    };
    edge.features[GraphFeature::ROUTE] = 0.0; // this edge is the turn that is not part of the route
    edge.features[GraphFeature::TYPE] = WayTypeId(way->GetType()->GetName());
    if (context.GetNodeId(*prev) != from.GetId()) {
      double inBearing = GetSphericalBearingFinal(prev->GetLocation(), from.GetCoord()).AsDegrees();
      double outBearing = GetSphericalBearingInitial(from.GetCoord(), to.GetCoord()).AsDegrees();
      double turnAngle = NormalizeRelativeAngle(outBearing - inBearing);
      edge.features[GraphFeature::ANGLE] = turnAngle;
    }
    if (laneDesc) {
      edge.features[GraphFeature::LANE_COUNT] = laneDesc->GetForwardLanes();
      for (size_t j = 0; j < laneDesc->GetTurnForward().size(); ++j) {
        edge.features["laneTurn" + std::to_string(j)] = static_cast<double>(laneDesc->GetTurnForward()[j]);
      }
    }
    if (accessDesc) {
      edge.features[GraphFeature::ONEWAY] = accessDesc->IsOneway() ? 1.0 : 0.0;
    }
    graph.edges.push_back(edge);
    distance += edge.length;
    if (distance > Meters(30)) {
      break; // Stop if the distance exceeds 50 meters
    }
  }
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
      auto fromNode = junctionStart;
      auto prevNode = junctionStart;
      graph.nodes.push_back(CreateGraphNode(context, *fromNode));
      for (auto toNode = std::next(junctionStart);
           toNode != end && distanceAhead < Meters(50) && toNode->GetPathObject().Valid();
           ++toNode) {
        assert(fromNode != toNode);

        // Create a graph node
        graph.InsertNode(CreateGraphNode(context, *toNode));

        // Create an edge
        GraphEdge edge=MakeEdge(context, prevNode, fromNode, toNode);
        if (ahead) {
          distanceAhead += edge.length;
        } else if (toNode == nodeIt) {
          ahead = true;
        }

        graph.edges.push_back(edge);

        for (const auto nodeExitRef: fromNode->GetObjects()){
          if (!nodeExitRef.Valid() ||
              !nodeExitRef.IsWay() ||
              nodeExitRef == fromNode->GetPathObject() ||
              nodeExitRef == prevNode->GetPathObject()) {
            continue;
          }
          auto nodeExit = context.GetWay(DBFileOffset(fromNode->GetDatabaseId(), nodeExitRef.GetFileOffset()));
          size_t intersectionId = std::numeric_limits<size_t>::max();
          for (size_t i = 0; i < nodeExit->nodes.size(); ++i) {
            if (nodeExit->nodes[i].GetId() == context.GetNodeId(*fromNode)) {
              intersectionId = i;
              break;
            }
          }
          assert(intersectionId != std::numeric_limits<size_t>::max());
          if (intersectionId > 0){
            TraverseWay(context, graph, fromNode->GetDatabaseId(), prevNode, nodeExit, intersectionId, -1);
          }
          if (intersectionId +1 < nodeExit->nodes.size()) {
            TraverseWay(context, graph, fromNode->GetDatabaseId(), prevNode, nodeExit, intersectionId, +1);
          }
        }

        prevNode = fromNode;
        fromNode = toNode;
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
