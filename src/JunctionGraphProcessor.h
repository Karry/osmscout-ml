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

#pragma once

#include <filesystem>

#include <osmscout/routing/RoutePostprocessor.h>

namespace osmscout {

struct GraphNode {
  Id id;
  GeoCoord location;
  GeoCoord normalizedLocation;
  int incoming=0;
  int outgoing=0;
};

namespace GraphFeature{
constexpr int EdgeFeatureCount = 9; // Update when adding new features, also update it in python code

// Per-highway features (same for all lane edges from same highway)
inline const std::string LANE_COUNT = "laneCount";
inline const std::string ANGLE = "angle";
inline const std::string ROUTE = "route"; // edge is part of the route
inline const std::string TYPE = "type";
inline const std::string USABLE = "usable"; // edge is usable by current vehicle
inline const std::string VIRTUAL = "virtual"; // edge is virtual (just for information propagation in GNN)

// Per-lane features (specific to each lane edge)
inline const std::string RELATIVE_LANE_POSITION = "relativeLanePosition"; // 0.0 (left) to 1.0 (right)
inline const std::string LANE_TURN = "laneTurn"; // turn direction for this specific lane
inline const std::string SUGGESTED = "suggested"; // binary: is this lane suggested for the route?

std::string WayTypeName(int typeId);
}

struct GraphEdge {
  Id fromNode;
  Id toNode;
  Distance length;
  std::unordered_map<std::string, double> features;

  bool isUsable() const;
  bool isVirtual() const;
};

struct Graph {
  std::vector<GraphNode> nodes;
  std::vector<GraphEdge> edges;

  std::set<Id> nodeIdSet;

  void Export(const std::filesystem::path &filePath) const;

  void Normalize();

  inline bool InsertNode(GraphNode node) {
    if (nodeIdSet.find(node.id) != nodeIdSet.end()) {
      return false;
    }
    nodes.push_back(node);
    nodeIdSet.insert(node.id);
    return true;
  }
};


class JunctionGraphProcessor: public RoutePostprocessor::Postprocessor
{
public:
  JunctionGraphProcessor() = default;
  ~JunctionGraphProcessor() override = default;

  JunctionGraphProcessor(const JunctionGraphProcessor&) = delete;
  JunctionGraphProcessor& operator=(const JunctionGraphProcessor&) = delete;

  JunctionGraphProcessor(JunctionGraphProcessor&&) = delete;
  JunctionGraphProcessor& operator=(JunctionGraphProcessor&&) = delete;

  bool Process(const PostprocessorContext& context,
               RouteDescription& description) override;

  virtual void ProcessJunctionGraph(const Graph &graph,
                                    const RouteDescription::Node &node);

};

class JunctionGraphExportProcessor: public JunctionGraphProcessor {
private:
  std::filesystem::path exportDirectory;

public:
  explicit JunctionGraphExportProcessor(const std::filesystem::path& exportDirectory);
  ~JunctionGraphExportProcessor() override = default;

  JunctionGraphExportProcessor(const JunctionGraphExportProcessor&) = delete;
  JunctionGraphExportProcessor& operator=(const JunctionGraphExportProcessor&) = delete;

  JunctionGraphExportProcessor(JunctionGraphExportProcessor&&) = delete;
  JunctionGraphExportProcessor& operator=(JunctionGraphExportProcessor&&) = delete;

  void ProcessJunctionGraph(const Graph &graph,
                            const RouteDescription::Node &node) override;
};

using JunctionGraphExportProcessorRef = std::shared_ptr<JunctionGraphExportProcessor>;

class ComplexJunctionGraphExportProcessor: public JunctionGraphExportProcessor {
public:
  explicit ComplexJunctionGraphExportProcessor(const std::filesystem::path& exportDirectory):
    JunctionGraphExportProcessor(exportDirectory)
  {}
  ~ComplexJunctionGraphExportProcessor() override = default;

  ComplexJunctionGraphExportProcessor(const ComplexJunctionGraphExportProcessor&) = delete;
  ComplexJunctionGraphExportProcessor& operator=(const ComplexJunctionGraphExportProcessor&) = delete;

  ComplexJunctionGraphExportProcessor(ComplexJunctionGraphExportProcessor&&) = delete;
  ComplexJunctionGraphExportProcessor& operator=(ComplexJunctionGraphExportProcessor&&) = delete;

  void ProcessJunctionGraph(const Graph &graph,
                            const RouteDescription::Node &node) override;

};
}
