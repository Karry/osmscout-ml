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

void Graph::Normalize()
{
  // evaluate angle of the first edge, then normalize all nodes
  // the way that the first edge heads to east (bearing 0) and has normalizedLocation (0,0)
  // all other nodes has normalizedLocation relative to this transformation

  if (nodes.empty() || edges.empty()) {
    return;
  }

  // Find the first edge to use as reference
  const auto& firstEdge = edges[0];

  // Find the nodes corresponding to the first edge
  auto fromNodeIt = std::find_if(nodes.begin(), nodes.end(),
    [&firstEdge](const GraphNode& node) { return node.id == firstEdge.fromNode; });
  auto toNodeIt = std::find_if(nodes.begin(), nodes.end(),
    [&firstEdge](const GraphNode& node) { return node.id == firstEdge.toNode; });

  if (fromNodeIt == nodes.end() || toNodeIt == nodes.end()) {
    return;
  }

  // Calculate the bearing of the first edge
  double bearing = GetSphericalBearingInitial(fromNodeIt->location, toNodeIt->location).AsDegrees();

  // Calculate rotation angle to make the first edge point east (0 degrees)
  double rotationAngle = -bearing;

  // Convert to radians for trigonometric functions
  double rotationRad = rotationAngle * M_PI / 180.0;
  double cosRot = std::cos(rotationRad);
  double sinRot = std::sin(rotationRad);

  // Use the first node as origin (0,0)
  GeoCoord origin = fromNodeIt->location;

  // Transform all nodes
  for (auto& node : nodes) {
    // Calculate relative position in meters using approximate local projection
    double deltaLat = (node.location.GetLat() - origin.GetLat()) * 111320.0; // meters per degree lat
    double deltaLon = (node.location.GetLon() - origin.GetLon()) * 111320.0 * std::cos(origin.GetLat() * M_PI / 180.0); // meters per degree lon

    // Apply rotation
    double rotatedX = deltaLon * cosRot - deltaLat * sinRot;
    double rotatedY = deltaLon * sinRot + deltaLat * cosRot;

    // Convert back to lat/lon for normalizedLocation (using approximate conversion)
    double normalizedLat = origin.GetLat() + rotatedY / 111320.0;
    double normalizedLon = origin.GetLon() + rotatedX / (111320.0 * std::cos(origin.GetLat() * M_PI / 180.0));

    node.normalizedLocation = GeoCoord(normalizedLat, normalizedLon);
  }

  // Adjust so that the first node is at (0,0)
  GeoCoord firstNodeNormalized = fromNodeIt->normalizedLocation;
  for (auto& node : nodes) {
    double adjustedLat = node.normalizedLocation.GetLat() - firstNodeNormalized.GetLat();
    double adjustedLon = node.normalizedLocation.GetLon() - firstNodeNormalized.GetLon();
    node.normalizedLocation = GeoCoord(adjustedLat, adjustedLon);
  }
}

void Graph::Export(const std::filesystem::path &filePath) const {
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
      {"lon", node.location.GetLon()},
      {"normLat", node.normalizedLocation.GetLat()},
      {"normLon", node.normalizedLocation.GetLon()}
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

namespace GraphFeature {
static const std::vector<std::string> wayTypes = {
  "highway_motorway",
  "highway_motorway_link",
  "highway_motorway_trunk",
  "highway_tertiary",
  "highway_trunk_link",
  "highway_residential",
  "highway_secondary",
  "highway_secondary_link",
  "highway_service",
  "highway_trunk",
  "highway_primary",
  "highway_primary_link",
  "highway_footway",
  "highway_track",
  "highway_tertiary_link",
  "highway_pedestrian",
  "highway_path",
  "highway_cycleway",
  "highway_via_ferrata_easy",
  "highway_via_ferrata_moderate",
  "highway_via_ferrata_difficult",
  "highway_via_ferrata_extreme",
  "highway_bridleway",
  "highway_steps",
  "highway_services",
  "highway_construction",
  "highway_roundabout",
  "highway_unclassified",
  "highway_road",
  "highway_living_street"
};


int WayTypeId(const std::string& typeName) {
  int index = 0;
  for (const auto& wayType : wayTypes) {
    if (wayType == typeName) {
      return index;
    }
    index++;
  }

  log.Warn() << "Unknown way type: " << typeName;
  return -1; // Unknown type
}

std::string WayTypeName(int typeId) {
  if (typeId < 0 || static_cast<size_t>(typeId) >= wayTypes.size()) {
    return "unknown";
  }
  return wayTypes[typeId];
}
}


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

std::pair<GraphEdge, GraphEdge> MakeEdge(const PostprocessorContext& context,
                                         const NodeIterator prev,
                                         const NodeIterator from,
                                         const NodeIterator to) {
  GraphEdge edge{
    context.GetNodeId(*from),
    context.GetNodeId(*to),
    GetSphericalDistance(from->GetLocation(), to->GetLocation())
  };
  edge.features[GraphFeature::ROUTE] = 1.0; // Mark this edge as part of the route
  edge.features[GraphFeature::USABLE] = 1.0; // edge should be usable when it is part of the route
  edge.features[GraphFeature::VIRTUAL] = 0.0; // edge is not virtual
  if (from->GetPathObject().IsWay()) {
    edge.features[GraphFeature::TYPE] = GraphFeature::WayTypeId(context.GetWay(from->GetDBFileOffset())->GetType()->GetName());
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

  GraphEdge reverse{
    context.GetNodeId(*to),
    context.GetNodeId(*from),
    edge.length
  };
  reverse.features[GraphFeature::VIRTUAL] = 1.0; // edge is virtual
  return {edge, reverse};
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
    edge.features[GraphFeature::VIRTUAL] = 0.0; // edge is not virtual
    edge.features[GraphFeature::ROUTE] = 0.0; // this edge is the turn that is not part of the route
    edge.features[GraphFeature::TYPE] = GraphFeature::WayTypeId(way->GetType()->GetName());
    if (direction < 0){
      edge.features[GraphFeature::USABLE] = context.CanUseBackward(dbId,
                                                                   way->GetId(id),
                                                                   way->GetObjectFileRef());
    } else {
      edge.features[GraphFeature::USABLE] = context.CanUseForward(dbId,
                                                                  way->GetId(id),
                                                                  way->GetObjectFileRef());
    }

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

    GraphEdge reverse{edge.toNode, edge.fromNode, edge.length};
    reverse.features[GraphFeature::VIRTUAL] = 1.0; // edge is virtual
    graph.edges.push_back(reverse);

    distance += edge.length;
    if (distance > Meters(30)) {
      break; // Stop if the distance exceeds 50 meters
    }
  }
}

} // anonymous namespace


void JunctionGraphProcessor::ProcessJunctionGraph(const Graph &graph,
                                                  const RouteDescription::Node &node)
{
  // Default implementation does nothing
  // Override this method to implement custom processing of the junction graph
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
        auto [edge, reverse]=MakeEdge(context, prevNode, fromNode, toNode);
        if (ahead) {
          distanceAhead += edge.length;
        } else if (toNode == nodeIt) {
          ahead = true;
        }

        graph.edges.push_back(edge);
        graph.edges.push_back(reverse);

        for (const auto nodeExitRef: fromNode->GetObjects()){
          if (!nodeExitRef.Valid() ||
              !nodeExitRef.IsWay() ||
              nodeExitRef == fromNode->GetPathObject() ||
              nodeExitRef == prevNode->GetPathObject()) {
            continue;
          }
          Id fromNodeId = context.GetNodeId(*fromNode);
          auto nodeExit = context.GetWay(DBFileOffset(fromNode->GetDatabaseId(), nodeExitRef.GetFileOffset()));

          size_t intersectionId;
          [[maybe_unused]] bool found=nodeExit->GetNodeIndexByNodeId(fromNodeId, intersectionId);
          assert(found);

          if (intersectionId > 0) {
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
        graph.Normalize();
        ProcessJunctionGraph(graph, node);
      }
      junctionStart = nodeIt;
    }
  }
  return true;
}

JunctionGraphExportProcessor::JunctionGraphExportProcessor(const std::filesystem::path& exportDirectory):
  JunctionGraphProcessor(),
  exportDirectory(exportDirectory)
{
  if (!std::filesystem::exists(exportDirectory)) {
    std::filesystem::create_directories(exportDirectory);
  }
  log.Debug() << "Junction graph export directory: " << exportDirectory;
}

void JunctionGraphExportProcessor::ProcessJunctionGraph(const Graph &graph,
                                                        const RouteDescription::Node &node)
{
  auto junctionFileName = std::to_string(node.GetPathObject().GetFileOffset()) + "_" + std::to_string(node.GetCurrentNodeIndex()) + ".json";
  log.Debug() << "Exporting junction graph for node "
              << node.GetPathObject().GetFileOffset() << "/" << node.GetCurrentNodeIndex()
              << " at " << node.GetLocation().GetDisplayText() << " to " << junctionFileName;
  graph.Export(exportDirectory / junctionFileName);
}

void ComplexJunctionGraphExportProcessor::ProcessJunctionGraph(const Graph &graph, const RouteDescription::Node &node)
{
  if (std::any_of(graph.edges.begin(), graph.edges.end(),
             [](const GraphEdge& edge) {
               return edge.features.contains(GraphFeature::LANE_COUNT) && edge.features.at(GraphFeature::LANE_COUNT) > 1.0;
             })) {
    JunctionGraphExportProcessor::ProcessJunctionGraph(graph, node);
  }
}

}
