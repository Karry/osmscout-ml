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

bool GraphEdge::isUsable() const
{
  return features.find(GraphFeature::USABLE)!=features.end() && features.at(GraphFeature::USABLE) > 0;
}

bool GraphEdge::isVirtual() const
{
  return features.find(GraphFeature::VIRTUAL)!=features.end() && features.at(GraphFeature::VIRTUAL) > 0;
}

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

  // compute cardinality (degree of each node)
  for (auto& node : nodes) {
    node.outgoing = 0;
    node.incoming = 0;
  }

  for (const auto& edge : edges) {
    if (!edge.isUsable() || edge.isVirtual()) {
      continue; // skip virtual or unusable edges
    }

    // Count outgoing edges for fromNode
    auto fromNodeIt = std::find_if(nodes.begin(), nodes.end(),
      [&edge](const GraphNode& node) { return node.id == edge.fromNode; });
    if (fromNodeIt != nodes.end()) {
      fromNodeIt->outgoing++;
    }

    // Count incoming edges for toNode
    auto toNodeIt = std::find_if(nodes.begin(), nodes.end(),
      [&edge](const GraphNode& node) { return node.id == edge.toNode; });
    if (toNodeIt != nodes.end()) {
      toNodeIt->incoming++;
    }
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
      {"normLon", node.normalizedLocation.GetLon()},
      {"incoming", node.incoming},
      {"outgoing", node.outgoing}
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

std::vector<GraphEdge> MakeLaneEdges(const PostprocessorContext& context,
                                     const NodeIterator prev,
                                     const NodeIterator from,
                                     const NodeIterator to) {
  std::vector<GraphEdge> edges;

  Distance length = GetSphericalDistance(from->GetLocation(), to->GetLocation());
  Id fromNodeId = context.GetNodeId(*from);
  Id toNodeId = context.GetNodeId(*to);

  // Calculate common features for all lane edges
  double turnAngle = 0.0;
  if (prev != from) {
    double inBearing = GetSphericalBearingFinal(prev->GetLocation(), from->GetLocation()).AsDegrees();
    double outBearing = GetSphericalBearingInitial(from->GetLocation(), to->GetLocation()).AsDegrees();
    turnAngle = NormalizeRelativeAngle(outBearing - inBearing);
  }

  double wayType = -1.0;
  if (from->GetPathObject().IsWay()) {
    wayType = GraphFeature::WayTypeId(context.GetWay(from->GetDBFileOffset())->GetType()->GetName());
  }

  // Get lane information
  auto laneDesc = from->GetDescription<RouteDescription::LaneDescription>();
  auto suggestedLanes = from->GetDescription<RouteDescription::SuggestedLaneDescription>();

  int laneCount = 1; // Default to single lane
  bool hasLaneInfo = false;

  if (laneDesc && laneDesc->GetLaneCount() > 0) {
    laneCount = laneDesc->GetLaneCount();
    hasLaneInfo = true;
  } else {
    // Missing lane information - print warning
    log.Warn() << "Missing lane information for way at node " << fromNodeId
               << " (" << from->GetLocation().GetDisplayText() << "), assuming single lane";
  }

  // Create one edge per lane
  for (int laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
    GraphEdge edge{fromNodeId, toNodeId, length};

    // Set common features (same for all lanes from this highway)
    edge.features[GraphFeature::LANE_COUNT] = static_cast<double>(laneCount);
    edge.features[GraphFeature::ANGLE] = turnAngle;
    edge.features[GraphFeature::ROUTE] = 1.0; // Mark this edge as part of the route
    edge.features[GraphFeature::TYPE] = wayType;
    edge.features[GraphFeature::USABLE] = 1.0; // edge should be usable when it is part of the route
    edge.features[GraphFeature::VIRTUAL] = 0.0; // edge is not virtual

    // Set lane-specific features
    // Calculate relative lane position: 0.0 (leftmost) to 1.0 (rightmost)
    edge.features[GraphFeature::RELATIVE_LANE_POSITION] =
      (laneCount > 1) ? static_cast<double>(laneIndex) / static_cast<double>(laneCount - 1) : 0.0;

    // Set lane turn direction
    if (hasLaneInfo && laneIndex < laneDesc->GetLaneTurns().size()) {
      edge.features[GraphFeature::LANE_TURN] = static_cast<double>(laneDesc->GetLaneTurns()[laneIndex]);
    } else {
      edge.features[GraphFeature::LANE_TURN] = static_cast<double>(LaneTurn::Unknown);
    }

    // Set suggested flag (binary: 1.0 if this lane is suggested, 0.0 otherwise)
    bool isSuggested = false;
    if (suggestedLanes) {
      int suggestedFrom = suggestedLanes->GetFrom();
      int suggestedTo = suggestedLanes->GetTo();
      isSuggested = (laneIndex >= suggestedFrom && laneIndex <= suggestedTo);
    }
    edge.features[GraphFeature::SUGGESTED] = isSuggested ? 1.0 : 0.0;

    edges.push_back(edge);
  }

  // Create one virtual reverse edge for the entire highway (not per lane)
  GraphEdge reverse{toNodeId, fromNodeId, length};
  reverse.features[GraphFeature::VIRTUAL] = 1.0; // edge is virtual
  edges.push_back(reverse);

  return edges;
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

    Distance length = GetSphericalDistance(from.GetCoord(), to.GetCoord());
    double wayType = GraphFeature::WayTypeId(way->GetType()->GetName());

    // Check if this edge is usable
    double isUsable = 0.0;
    if (direction < 0) {
      isUsable = context.CanUseBackward(dbId, way->GetId(id), way->GetObjectFileRef());
    } else {
      isUsable = context.CanUseForward(dbId, way->GetId(id), way->GetObjectFileRef());
    }

    // Calculate turn angle
    double turnAngle = 0.0;
    if (context.GetNodeId(*prev) != from.GetId()) {
      double inBearing = GetSphericalBearingFinal(prev->GetLocation(), from.GetCoord()).AsDegrees();
      double outBearing = GetSphericalBearingInitial(from.GetCoord(), to.GetCoord()).AsDegrees();
      turnAngle = NormalizeRelativeAngle(outBearing - inBearing);
    }

    // Get lane information
    int laneCount = 1; // Default to single lane
    std::vector<LaneTurn> laneTurns;
    bool hasLaneInfo = false;

    if (laneDesc) {
      laneCount = laneDesc->GetForwardLanes();
      laneTurns = laneDesc->GetTurnForward();
      hasLaneInfo = true;
    } else {
      // Missing lane information - print warning
      log.Warn() << "Missing lane information for way " << way->GetObjectFileRef().GetName()
                 << " at node " << from.GetId() << ", assuming single lane";
    }

    // Create one edge per lane
    for (int laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
      GraphEdge edge{from.GetId(), to.GetId(), length};

      // Set common features
      edge.features[GraphFeature::VIRTUAL] = 0.0;
      edge.features[GraphFeature::ROUTE] = 0.0; // this edge is not part of the route
      edge.features[GraphFeature::TYPE] = wayType;
      edge.features[GraphFeature::USABLE] = isUsable;
      edge.features[GraphFeature::LANE_COUNT] = static_cast<double>(laneCount);
      edge.features[GraphFeature::ANGLE] = turnAngle;

      // Set lane-specific features
      edge.features[GraphFeature::RELATIVE_LANE_POSITION] =
        (laneCount > 1) ? static_cast<double>(laneIndex) / static_cast<double>(laneCount - 1) : 0.0;

      if (hasLaneInfo && laneIndex < laneTurns.size()) {
        edge.features[GraphFeature::LANE_TURN] = static_cast<double>(laneTurns[laneIndex]);
      } else {
        edge.features[GraphFeature::LANE_TURN] = static_cast<double>(LaneTurn::Unknown);
      }

      // Not part of route, so never suggested
      edge.features[GraphFeature::SUGGESTED] = 0.0;

      graph.edges.push_back(edge);
    }

    // Create one virtual reverse edge for the entire highway
    GraphEdge reverse{to.GetId(), from.GetId(), length};
    reverse.features[GraphFeature::VIRTUAL] = 1.0;
    graph.edges.push_back(reverse);

    distance += length;
    if (distance > Meters(30)) {
      break; // Stop if the distance exceeds 30 meters
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

        // Create lane edges (one per lane + one virtual reverse)
        auto laneEdges = MakeLaneEdges(context, prevNode, fromNode, toNode);
        if (ahead && !laneEdges.empty()) {
          distanceAhead += laneEdges[0].length;
        } else if (toNode == nodeIt) {
          ahead = true;
        }

        // Add all lane edges to the graph
        for (auto& edge : laneEdges) {
          graph.edges.push_back(edge);
        }

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
