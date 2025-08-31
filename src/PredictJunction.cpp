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

#include <osmscout/db/Database.h>

#include <osmscout/routing/SimpleRoutingService.h>
#include <osmscout/routing/RoutePostprocessor.h>
#include <osmscout/routing/DBFileOffset.h>
#include <osmscout/routing/RouteDescriptionPostprocessor.h>

#include <osmscout/cli/CmdLineParsing.h>
#include <osmscout/util/Bearing.h>
#include <osmscout/util/Geometry.h>

#include <JunctionGraphProcessor.h>

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
  std::filesystem::path             junctionExportDir=std::filesystem::current_path();
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

static void GetCarSpeedTable(std::map<std::string,double>& map)
{
  map["highway_motorway"]=110.0;
  map["highway_motorway_trunk"]=100.0;
  map["highway_motorway_primary"]=70.0;
  map["highway_motorway_link"]=60.0;
  map["highway_motorway_junction"]=60.0;
  map["highway_trunk"]=100.0;
  map["highway_trunk_link"]=60.0;
  map["highway_primary"]=70.0;
  map["highway_primary_link"]=60.0;
  map["highway_secondary"]=60.0;
  map["highway_secondary_link"]=50.0;
  map["highway_tertiary_link"]=55.0;
  map["highway_tertiary"]=55.0;
  map["highway_unclassified"]=50.0;
  map["highway_road"]=50.0;
  map["highway_residential"]=20.0;
  map["highway_roundabout"]=40.0;
  map["highway_living_street"]=10.0;
  map["highway_service"]=30.0;
}

class ConsoleRoutingProgress : public osmscout::RoutingProgress
{
private:
  std::chrono::system_clock::time_point lastDump=std::chrono::system_clock::now();
  double                                maxPercent=0.0;

public:
  ConsoleRoutingProgress() = default;

  void Reset() override
  {
    lastDump=std::chrono::system_clock::now();
    maxPercent=0.0;
  }

  void Progress(const osmscout::Distance &currentMaxDistance,
                const osmscout::Distance &overallDistance) override
  {
    double currentPercent=(currentMaxDistance.AsMeter()*100.0)/overallDistance.AsMeter();

    std::chrono::system_clock::time_point now=std::chrono::system_clock::now();

    maxPercent=std::max(maxPercent,currentPercent);

    if (std::chrono::duration_cast<std::chrono::milliseconds>(now-lastDump).count()>500) {
      std::cout << "Progress: " << (size_t)maxPercent << "%" << std::endl;

      lastDump=now;
    }
  }
};

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
                        args.junctionExportDir=value;
                      }),
                      "junctionExportDir",
                      "Directory for exporting junction JSON files",
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

  std::cout << "Model path: " << args.modelPath << std::endl;

  torch::jit::script::Module model;
  try {
    model = torch::jit::load(args.modelPath);
  } catch (const std::exception& e) {
    std::cerr << "Error loading model: " << e.what() << std::endl;
    return 1;
  }

  // TODO: Implement the rest of the junction prediction logic
  // This will include:
  // 1. Database and routing setup (similar to JunctionGraphExport)
  // 2. Route calculation
  // 3. Junction detection along the route
  // 4. Model inference for junction predictions
  // 5. Output of predicted lane suggestions

  std::cout << "PredictJunction utility - Implementation pending" << std::endl;
  std::cout << "Database: " << args.databaseDirectory << std::endl;
  std::cout << "Start: " << args.start.GetDisplayText() << std::endl;
  std::cout << "Target: " << args.target.GetDisplayText() << std::endl;

  return 0;
}
