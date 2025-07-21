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

#include <JunctionGraphProcessor.h>

namespace osmscout {

using NodeIterator = std::list<RouteDescription::Node>::iterator;

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
}

struct GraphNode {
  Id id;
  GeoCoord location;
};

struct GraphEdge {
  Id fromNode;
  Id toNode;
  Distance length;
};

struct Graph {
  std::vector<GraphNode> nodes;
  std::vector<GraphEdge> edges;
};


bool JunctionGraphProcessor::Process(const PostprocessorContext& context,
                                     RouteDescription& description) {

  auto junctionStart = description.Nodes().begin();
  for (auto nodeIt = description.Nodes().begin();
       nodeIt != description.Nodes().end();
       ++nodeIt) {
    auto& node = *nodeIt;
    while (std::distance(nodeIt, junctionStart) > 1 &&
           SegmentLength(junctionStart, nodeIt) > Meters(50)) {
      assert(junctionStart != nodeIt);
      ++junctionStart;
    }

    if (node.HasDescription(RouteDescription::TURN_DESC) ||
        node.HasDescription(RouteDescription::MOTORWAY_CHANGE_DESC) ||
        node.HasDescription(RouteDescription::MOTORWAY_LEAVE_DESC) ||
        node.HasDescription(RouteDescription::MOTORWAY_JUNCTION_DESC)
        ) {
        log.Debug() << "Exporting junction graph for node "
                    << node.GetPathObject().GetFileOffset() << " / " << node.GetCurrentNodeIndex()
                    << " at " << node.GetLocation().GetDisplayText();

        Graph graph;
        // TODO: Fill the graph with nodes and edges

        junctionStart = nodeIt;
    }
  }
  return true;
}

}
