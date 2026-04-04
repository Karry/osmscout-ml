/*
  OSMScout ML - Junction Map Visualization Tool
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
#include <unordered_map>
#include <cmath>

#include <osmscout/db/Database.h>
#include <osmscout/projection/MercatorProjection.h>
#include <osmscout/cli/CmdLineParsing.h>
#include <osmscout/util/GeoBox.h>

#include <osmscoutmap/MapService.h>
#include <osmscoutmapqt/MapPainterQt.h>

#include <QGuiApplication>
#include <QPixmap>
#include <QPainter>

#include <JunctionGraphProcessor.h>

#ifdef HAS_TORCH
#include <torch/torch.h>
#include <torch/script.h>
#endif

struct Arguments {
  bool help = false;
  std::string junctionJson;
  std::string databaseDirectory;
  std::string stylePath;
  std::string modelPath;
  std::string outputDir;
  size_t width = 1024;
  size_t height = 1024;
  unsigned int zoom = 18;
  double dpi = 96.0;
};

enum class OverlayMode {
  Heuristic,
  MLPrediction
};

// Find a graph node by its ID
const osmscout::GraphNode* FindNode(const osmscout::Graph& graph, osmscout::Id nodeId) {
  for (const auto& node : graph.nodes) {
    if (node.id == nodeId) {
      return &node;
    }
  }
  return nullptr;
}

// Draw the junction graph overlay onto a QPainter
void DrawJunctionOverlay(QPainter& painter,
                         const osmscout::MercatorProjection& projection,
                         const osmscout::Graph& graph,
                         OverlayMode mode,
                         const std::vector<float>* predictions = nullptr) {
  painter.setRenderHint(QPainter::Antialiasing);

  constexpr double laneWidthPx = 5.0;

  // Draw edges
  for (size_t i = 0; i < graph.edges.size(); ++i) {
    const auto& edge = graph.edges[i];

    // Skip virtual edges
    if (edge.isVirtual()) {
      continue;
    }

    const auto* fromNode = FindNode(graph, edge.fromNode);
    const auto* toNode = FindNode(graph, edge.toNode);
    if (!fromNode || !toNode) {
      continue;
    }

    // Convert geo coordinates to pixel coordinates
    osmscout::Vertex2D fromPixel, toPixel;
    if (!projection.GeoToPixel(fromNode->location, fromPixel) ||
        !projection.GeoToPixel(toNode->location, toPixel)) {
      continue;
    }

    // Compute perpendicular offset for multi-lane display
    double dx = toPixel.GetX() - fromPixel.GetX();
    double dy = toPixel.GetY() - fromPixel.GetY();
    double len = std::sqrt(dx * dx + dy * dy);

    double offsetX = 0.0, offsetY = 0.0;
    if (len > 0.1) {
      double nx = -dy / len;
      double ny = dx / len;

      double laneCount = edge.features.count(osmscout::GraphFeature::LANE_COUNT)
                         ? edge.features.at(osmscout::GraphFeature::LANE_COUNT) : 1.0;
      double relPos = edge.features.count(osmscout::GraphFeature::RELATIVE_LANE_POSITION)
                      ? edge.features.at(osmscout::GraphFeature::RELATIVE_LANE_POSITION) : 0.0;

      if (laneCount > 1) {
        double totalWidth = laneWidthPx * laneCount;
        double offset = (relPos - 0.5) * totalWidth;
        offsetX = nx * offset;
        offsetY = ny * offset;
      }
    }

    double x1 = fromPixel.GetX() + offsetX;
    double y1 = fromPixel.GetY() + offsetY;
    double x2 = toPixel.GetX() + offsetX;
    double y2 = toPixel.GetY() + offsetY;

    // Determine color based on overlay mode
    QColor color;
    if (mode == OverlayMode::Heuristic) {
      bool suggested = edge.features.count(osmscout::GraphFeature::SUGGESTED) &&
                       edge.features.at(osmscout::GraphFeature::SUGGESTED) > 0.5;
      color = suggested ? QColor(0, 200, 0, 180) : QColor(200, 0, 0, 180);
    } else {
      // ML prediction mode
      if (predictions && i < predictions->size()) {
        float pred = (*predictions)[i];
        int r = static_cast<int>((1.0f - pred) * 200);
        int g = static_cast<int>(pred * 200);
        int alpha = static_cast<int>(100 + 80 * std::abs(pred - 0.5f) * 2.0f);
        color = QColor(r, g, 0, alpha);
      } else {
        color = QColor(128, 128, 128, 100); // gray for edges without prediction
      }
    }

    // Route edges are thicker
    bool isRoute = edge.features.count(osmscout::GraphFeature::ROUTE) &&
                   edge.features.at(osmscout::GraphFeature::ROUTE) > 0.5;
    double lineWidth = isRoute ? 4.0 : 2.0;

    QPen pen(color);
    pen.setWidthF(lineWidth);
    pen.setCapStyle(Qt::RoundCap);
    painter.setPen(pen);
    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2));
  }

  // Draw node markers
  for (const auto& node : graph.nodes) {
    osmscout::Vertex2D pixel;
    if (!projection.GeoToPixel(node.location, pixel)) {
      continue;
    }
    QPointF pos(pixel.GetX(), pixel.GetY());

    if (graph.junctionStart && node.id == *graph.junctionStart) {
      // Junction entry: green dot
      painter.setPen(QPen(QColor(0, 0, 0), 1.5));
      painter.setBrush(QColor(0, 200, 0, 220));
      painter.drawEllipse(pos, 6.0, 6.0);
    } else if (graph.junctionEnd && node.id == *graph.junctionEnd) {
      // Junction exit: red dot
      painter.setPen(QPen(QColor(0, 0, 0), 1.5));
      painter.setBrush(QColor(200, 0, 0, 220));
      painter.drawEllipse(pos, 6.0, 6.0);
    } else {
      painter.setPen(Qt::NoPen);
      painter.setBrush(QColor(50, 50, 50, 200));
      painter.drawEllipse(pos, 3.0, 3.0);
    }
  }

  // Draw legend
  int legendX = 10;
  int legendY = 20;
  painter.setFont(QFont("sans-serif", 10));

  auto drawLegendEntry = [&](const QColor& c, const QString& text) {
    painter.setPen(Qt::NoPen);
    painter.setBrush(c);
    painter.drawRect(legendX, legendY - 10, 14, 14);
    painter.setPen(QColor(0, 0, 0));
    painter.drawText(legendX + 18, legendY + 2, text);
    legendY += 20;
  };

  if (mode == OverlayMode::Heuristic) {
    drawLegendEntry(QColor(0, 200, 0, 180), "Suggested");
    drawLegendEntry(QColor(200, 0, 0, 180), "Not suggested");
  } else {
    drawLegendEntry(QColor(0, 200, 0, 180), "Predicted");
    drawLegendEntry(QColor(200, 0, 0, 180), "Not predicted");
  }
  if (graph.junctionStart || graph.junctionEnd) {
    drawLegendEntry(QColor(0, 200, 0, 220), "Junction entry");
    drawLegendEntry(QColor(200, 0, 0, 220), "Junction exit");
  }
}

#ifdef HAS_TORCH
// Run ML inference on a junction graph, returning per-edge sigmoid probabilities
std::vector<float> RunInference(const osmscout::Graph& graph,
                                torch::jit::script::Module& model) {
  std::vector<float> result(graph.edges.size(), 0.0f);

  std::unordered_map<osmscout::Id, int> nodeIdToIndex;
  for (size_t i = 0; i < graph.nodes.size(); ++i) {
    nodeIdToIndex[graph.nodes[i].id] = static_cast<int>(i);
  }

  // Node features [N, 4]
  const int nodeFeatureCount = 4;
  torch::Tensor nodeTensor = torch::zeros({static_cast<int64_t>(graph.nodes.size()), nodeFeatureCount});
  for (size_t i = 0; i < graph.nodes.size(); ++i) {
    nodeTensor[i][0] = static_cast<float>(graph.nodes[i].normalizedLocation.GetLat());
    nodeTensor[i][1] = static_cast<float>(graph.nodes[i].normalizedLocation.GetLon());
    nodeTensor[i][2] = static_cast<float>(graph.nodes[i].incoming);
    nodeTensor[i][3] = static_cast<float>(graph.nodes[i].outgoing);
  }

  // Edge index [2, E] and edge attributes [E, 9]
  std::vector<std::array<int64_t, 2>> edgeIndices;
  std::vector<std::array<float, osmscout::GraphFeature::EdgeFeatureCount>> edgeFeatures;

  for (const auto& edge : graph.edges) {
    auto fromIt = nodeIdToIndex.find(edge.fromNode);
    auto toIt = nodeIdToIndex.find(edge.toNode);
    if (fromIt == nodeIdToIndex.end() || toIt == nodeIdToIndex.end()) {
      continue;
    }

    edgeIndices.push_back({fromIt->second, toIt->second});

    std::array<float, osmscout::GraphFeature::EdgeFeatureCount> feats{};
    auto getF = [&](const std::string& key, float def) -> float {
      auto it = edge.features.find(key);
      return it != edge.features.end() ? static_cast<float>(it->second) : def;
    };
    feats[0] = static_cast<float>(edge.length.AsMeter());
    feats[1] = getF(osmscout::GraphFeature::LANE_COUNT, 0.0f);
    feats[2] = getF(osmscout::GraphFeature::ANGLE, 0.0f);
    feats[3] = getF(osmscout::GraphFeature::ROUTE, 0.0f);
    feats[4] = getF(osmscout::GraphFeature::TYPE, -1.0f);
    feats[5] = getF(osmscout::GraphFeature::USABLE, 0.0f);
    feats[6] = getF(osmscout::GraphFeature::VIRTUAL, 0.0f);
    feats[7] = getF(osmscout::GraphFeature::RELATIVE_LANE_POSITION, 0.0f);
    feats[8] = getF(osmscout::GraphFeature::LANE_TURN, -1.0f);
    edgeFeatures.push_back(feats);
  }

  if (edgeIndices.empty()) {
    return result;
  }

  auto E = static_cast<int64_t>(edgeIndices.size());

  torch::Tensor edgeIndexTensor = torch::zeros({2, E}, torch::kInt64);
  for (int64_t i = 0; i < E; ++i) {
    edgeIndexTensor[0][i] = edgeIndices[i][0];
    edgeIndexTensor[1][i] = edgeIndices[i][1];
  }

  torch::Tensor edgeAttrTensor = torch::zeros({E, osmscout::GraphFeature::EdgeFeatureCount});
  for (int64_t i = 0; i < E; ++i) {
    for (int j = 0; j < osmscout::GraphFeature::EdgeFeatureCount; ++j) {
      edgeAttrTensor[i][j] = edgeFeatures[i][j];
    }
  }

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(nodeTensor);
  inputs.push_back(edgeIndexTensor);
  inputs.push_back(edgeAttrTensor);

  torch::jit::IValue output = model.forward(inputs);
  torch::Tensor predictions = torch::sigmoid(output.toTensor());

  // Map predictions back to graph edge indices
  // edgeIndices was built by iterating graph.edges in order, skipping edges where
  // node lookup failed. We need to map back carefully.
  size_t predIdx = 0;
  for (size_t i = 0; i < graph.edges.size(); ++i) {
    const auto& edge = graph.edges[i];
    auto fromIt = nodeIdToIndex.find(edge.fromNode);
    auto toIt = nodeIdToIndex.find(edge.toNode);
    if (fromIt != nodeIdToIndex.end() && toIt != nodeIdToIndex.end()) {
      if (predIdx < static_cast<size_t>(predictions.size(0))) {
        result[i] = predictions[predIdx].item<float>();
      }
      predIdx++;
    }
  }

  return result;
}
#endif

int main(int argc, char* argv[]) {
  using namespace osmscout;

  // Parse CLI arguments before creating QGuiApplication (which modifies argc/argv)
  CmdLineParser argParser("DrawJunction", argc, argv);
  Arguments args;

  argParser.AddOption(CmdLineFlag([&args](const bool& value) {
    args.help = value;
  }), std::vector<std::string>{"h", "help"}, "Display help", true);

  argParser.AddOption(CmdLineStringOption([&args](const std::string& value) {
    args.databaseDirectory = value;
  }), "database", "OSM database directory", false);

  argParser.AddOption(CmdLineStringOption([&args](const std::string& value) {
    args.stylePath = value;
  }), "style", "Stylesheet path (.oss)", false);

  argParser.AddOption(CmdLineStringOption([&args](const std::string& value) {
    args.modelPath = value;
  }), "model", "PyTorch TorchScript model path", false);

  argParser.AddOption(CmdLineStringOption([&args](const std::string& value) {
    args.outputDir = value;
  }), "output", "Output directory for PNGs", false);

  argParser.AddOption(CmdLineSizeTOption([&args](const size_t& value) {
    args.width = value;
  }), "width", "Image width in pixels (default: 1024)");

  argParser.AddOption(CmdLineSizeTOption([&args](const size_t& value) {
    args.height = value;
  }), "height", "Image height in pixels (default: 1024)");

  argParser.AddOption(CmdLineUIntOption([&args](unsigned int value) {
    args.zoom = value;
  }), "zoom", "Magnification level (default: 18)");

  argParser.AddOption(CmdLineDoubleOption([&args](double value) {
    args.dpi = value;
  }), "dpi", "Rendering DPI (default: 96.0)");

  argParser.AddPositional(CmdLineStringOption([&args](const std::string& value) {
    args.junctionJson = value;
  }), "JUNCTION_JSON", "Path to junction graph JSON file");

  auto parseResult = argParser.Parse();
  if (parseResult.HasError()) {
    std::cerr << "ERROR: " << parseResult.GetErrorDescription() << std::endl;
    std::cout << argParser.GetHelp() << std::endl;
    return 1;
  }

  if (args.help) {
    std::cout << argParser.GetHelp() << std::endl;
    return 0;
  }

  if (args.databaseDirectory.empty()) {
    std::cerr << "Error: --database is required" << std::endl;
    return 1;
  }
  if (args.stylePath.empty()) {
    std::cerr << "Error: --style is required" << std::endl;
    return 1;
  }

  // Initialize Qt offscreen rendering
  qputenv("QT_QPA_PLATFORM", "offscreen");
  QGuiApplication app(argc, argv);

  // Load junction graph
  std::filesystem::path jsonPath(args.junctionJson);
  if (!std::filesystem::exists(jsonPath)) {
    std::cerr << "Error: Junction JSON file does not exist: " << jsonPath << std::endl;
    return 1;
  }

  Graph graph;
  try {
    graph.Import(jsonPath);
  } catch (const std::exception& e) {
    std::cerr << "Error loading junction graph: " << e.what() << std::endl;
    return 1;
  }

  if (graph.nodes.empty() || graph.edges.empty()) {
    std::cerr << "Error: Junction graph is empty" << std::endl;
    return 1;
  }

  std::cout << "Loaded junction graph: " << graph.nodes.size() << " nodes, "
            << graph.edges.size() << " edges" << std::endl;

  // Compute junction center
  GeoBox bbox;
  for (const auto& node : graph.nodes) {
    bbox.Include(node.location);
  }
  GeoCoord center = bbox.GetCenter();
  std::cout << "Junction center: " << center.GetDisplayText() << std::endl;

  // Open OSM database
  DatabaseParameter databaseParameter;
  auto database = std::make_shared<Database>(databaseParameter);
  if (!database->Open(args.databaseDirectory)) {
    std::cerr << "Error: Cannot open database: " << args.databaseDirectory << std::endl;
    return 1;
  }

  auto styleConfig = std::make_shared<StyleConfig>(database->GetTypeConfig());
  if (!styleConfig->Load(args.stylePath)) {
    std::cerr << "Error: Cannot load stylesheet: " << args.stylePath << std::endl;
    return 1;
  }

  auto mapService = std::make_shared<MapService>(database);

  // Set up projection
  MercatorProjection projection;
  Magnification magnification;
  magnification.SetLevel(MagnificationLevel(args.zoom));

  projection.Set(center, 0.0, magnification, args.dpi, args.width, args.height);

  // Load map tile data
  AreaSearchParameter searchParameter;
  MapData mapData;
  mapData.styleConfig = styleConfig;

  std::list<TileRef> tiles;
  mapService->LookupTiles(projection, tiles);
  mapService->LoadMissingTileData(searchParameter, *styleConfig, tiles);
  mapService->AddTileDataToMapData(tiles, mapData);
  mapService->GetGroundTiles(projection, mapData.groundTiles);

  // Render base map
  QPixmap basePixmap(static_cast<int>(args.width), static_cast<int>(args.height));
  basePixmap.fill(Qt::white);

  {
    QPainter painter(&basePixmap);
    MapParameter drawParameter;
    drawParameter.SetFontName("/usr/share/fonts/TTF/LiberationSans-Regular.ttf");
    drawParameter.SetFontSize(3.0);
    drawParameter.SetRenderSeaLand(true);
    drawParameter.SetRenderUnknowns(false);
    drawParameter.SetRenderBackground(true);
    drawParameter.SetLabelLineMinCharCount(15);
    drawParameter.SetLabelLineMaxCharCount(30);
    drawParameter.SetLabelLineFitToArea(true);

    MapPainterQt mapPainter;
    std::vector<MapData> dataList;
    dataList.emplace_back(std::move(mapData));

    if (!mapPainter.DrawMap(projection, drawParameter, dataList, &painter)) {
      std::cerr << "Warning: DrawMap returned false" << std::endl;
    }
  }

  // Determine output paths
  std::filesystem::path outputDir = args.outputDir.empty()
    ? jsonPath.parent_path()
    : std::filesystem::path(args.outputDir);
  std::string stem = jsonPath.stem().string();

  if (!std::filesystem::exists(outputDir)) {
    std::filesystem::create_directories(outputDir);
  }

  // Draw heuristic overlay
  {
    QPixmap heuristicPixmap = basePixmap;
    QPainter painter(&heuristicPixmap);
    DrawJunctionOverlay(painter, projection, graph, OverlayMode::Heuristic);
    painter.end();

    auto outPath = outputDir / (stem + "_heuristic.png");
    if (heuristicPixmap.save(QString::fromStdString(outPath.string()), "PNG")) {
      std::cout << "Saved heuristic overlay: " << outPath << std::endl;
    } else {
      std::cerr << "Error: Failed to save " << outPath << std::endl;
    }
  }

  // Draw ML prediction overlay (if model provided)
#ifdef HAS_TORCH
  if (!args.modelPath.empty()) {
    try {
      auto model = torch::jit::load(args.modelPath);
      model.eval();

      std::cout << "Running ML inference..." << std::endl;
      auto predictions = RunInference(graph, model);

      QPixmap mlPixmap = basePixmap;
      QPainter painter(&mlPixmap);
      DrawJunctionOverlay(painter, projection, graph, OverlayMode::MLPrediction, &predictions);
      painter.end();

      auto outPath = outputDir / (stem + "_predicted.png");
      if (mlPixmap.save(QString::fromStdString(outPath.string()), "PNG")) {
        std::cout << "Saved ML prediction overlay: " << outPath << std::endl;
      } else {
        std::cerr << "Error: Failed to save " << outPath << std::endl;
      }
    } catch (const std::exception& e) {
      std::cerr << "Error during ML inference: " << e.what() << std::endl;
    }
  }
#else
  if (!args.modelPath.empty()) {
    std::cerr << "Warning: --model specified but DrawJunction was built without PyTorch support" << std::endl;
  }
#endif

  return 0;
}
