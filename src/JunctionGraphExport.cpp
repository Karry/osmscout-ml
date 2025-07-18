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
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <list>
#include <sstream>
#include <fstream>
#include <optional>

#include <osmscout/db/Database.h>

#include <osmscout/routing/SimpleRoutingService.h>
#include <osmscout/routing/RoutePostprocessor.h>
#include <osmscout/routing/DBFileOffset.h>
#include <osmscout/routing/RouteDescriptionPostprocessor.h>

#include <osmscout/cli/CmdLineParsing.h>
#include <osmscout/util/Bearing.h>
#include <osmscout/util/Geometry.h>
#include <osmscout/util/Time.h>

#include <JunctionGraphProcessor.h>

struct Arguments
{
  bool                              help=false;
  std::string                       router=osmscout::RoutingService::DEFAULT_FILENAME_BASE;
  osmscout::Vehicle                 vehicle=osmscout::Vehicle::vehicleCar;
  std::string                       gpx;
  std::string                       databaseDirectory;
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
      std::cout << (size_t)maxPercent << "%" << std::endl;

      lastDump=now;
    }
  }
};


int main(int argc, char* argv[]) {
  using namespace osmscout;
  using namespace std::string_literals;
  using namespace std::chrono;

  osmscout::CmdLineParser   argParser("Routing",
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
                      "Dump data nodes to sdt::cout",
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

  osmscout::log.Debug(args.debug);
  osmscout::log.Info(true);
  osmscout::log.Warn(true);
  osmscout::log.Error(true);

  osmscout::DatabaseParameter databaseParameter;
  osmscout::DatabaseRef       database=std::make_shared<osmscout::Database>(databaseParameter);

  if (!database->Open(args.databaseDirectory)) {
    std::cerr << "Cannot open db" << std::endl;

    return 1;
  }

  osmscout::FastestPathRoutingProfileRef routingProfile=std::make_shared<osmscout::FastestPathRoutingProfile>(database->GetTypeConfig());
  osmscout::RouterParameter              routerParameter;

  routingProfile->SetPenaltySameType(args.penaltySameType);
  routingProfile->SetPenaltyDifferentType(args.penaltyDifferentType);
  routingProfile->SetMaxPenalty(args.maxPenalty);

  routerParameter.SetDebugPerformance(true);

  osmscout::SimpleRoutingServiceRef router=std::make_shared<osmscout::SimpleRoutingService>(database,
                                                                                            routerParameter,
                                                                                            args.router);

  if (!router->Open()) {
    std::cerr << "Cannot open routing db" << std::endl;

    return 1;
  }

  osmscout::TypeConfigRef             typeConfig=database->GetTypeConfig();
  std::map<std::string,double>        carSpeedTable;
  osmscout::RoutingParameter          parameter;

  parameter.SetProgress(std::make_shared<ConsoleRoutingProgress>());

  switch (args.vehicle) {
    case osmscout::vehicleFoot:
      routingProfile->ParametrizeForFoot(*typeConfig,
                                         5.0);
      break;
    case osmscout::vehicleBicycle:
      routingProfile->ParametrizeForBicycle(*typeConfig,
                                            20.0);
      break;
    case osmscout::vehicleCar:
      GetCarSpeedTable(carSpeedTable);
      routingProfile->ParametrizeForCar(*typeConfig,
                                        carSpeedTable,
                                        160.0);
      break;
  }

  auto startResult=router->GetClosestRoutableNode(args.start,
                                                  *routingProfile,
                                                  osmscout::Kilometers(1));

  if (!startResult.IsValid()) {
    std::cerr << "Error while searching for routing node near start location!" << std::endl;
    return 1;
  }

  osmscout::RoutePosition start=startResult.GetRoutePosition();
  if (start.GetObjectFileRef().GetType()==osmscout::refNode) {
    std::cerr << "Cannot find start node for start location!" << std::endl;
  }

  auto targetResult=router->GetClosestRoutableNode(args.target,
                                                   *routingProfile,
                                                   osmscout::Kilometers(1));

  if (!targetResult.IsValid()) {
    std::cerr << "Error while searching for routing node near target location!" << std::endl;
    return 1;
  }

  osmscout::RoutePosition target=targetResult.GetRoutePosition();
  if (target.GetObjectFileRef().GetType()==osmscout::refNode) {
    std::cerr << "Cannot find start node for target location!" << std::endl;
  }

  osmscout::RoutingResult result;

  if (args.via.size() > 0) {
    std::cout << "Using 'CalculateRouteViaCoords' method" << std::endl;
    args.via.insert(args.via.begin(), args.start);
    args.via.push_back(args.target);
    result=router->CalculateRouteViaCoords(*routingProfile,
                                           args.via,
                                           osmscout::Kilometers(1),
                                           parameter);

  } else {
    std::cout << "Using 'CalculateRoute' method" << std::endl;
    result=router->CalculateRoute(*routingProfile,
                                  start,
                                  target,
                                  args.initialBearing,
                                  parameter);
  }

  if (!result.Success()) {
    std::cerr << "There was an error while calculating the route!" << std::endl;
    router->Close();
    return 1;
  }

  if (args.dataDebug) {
    std::cout << "Route raw data:" << std::endl;
    for (const auto &entry : result.GetRoute().Entries()) {
      std::cout << entry.GetPathObject().GetName() << "[" << entry.GetCurrentNodeIndex() << "]" << " = "
                << entry.GetCurrentNodeId() << " => " << entry.GetTargetNodeIndex() << std::endl;
    }
  }

  auto routeDescriptionResult=router->TransformRouteDataToRouteDescription(result.GetRoute());

  if (!routeDescriptionResult.Success()) {
    std::cerr << "Error during generation of route description" << std::endl;
    return 1;
  }

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
    std::make_shared<osmscout::JunctionGraphProcessor>(),
  };

  osmscout::RoutePostprocessor postprocessor;

  osmscout::StopClock postprocessTimer;

  std::set<std::string,std::less<>>        motorwayTypeNames{"highway_motorway",
                                                             "highway_motorway_trunk",
                                                             "highway_trunk",
                                                             "highway_motorway_primary"};
  std::set<std::string,std::less<>>        motorwayLinkTypeNames{"highway_motorway_link",
                                                                 "highway_trunk_link"};
  std::set<std::string,std::less<>>        junctionTypeNames{"highway_motorway_junction"};

  std::vector<osmscout::RoutingProfileRef> profiles{routingProfile};
  std::vector<osmscout::DatabaseRef>       databases{database};

  // SectionsPostprocessor needs the section lenghts computed in the routing when there are some via points
  // between start and end
  postprocessors.push_back(std::make_shared<osmscout::RoutePostprocessor::SectionsPostprocessor>(result.GetSectionLenghts()));

  if (!postprocessor.PostprocessRouteDescription(*routeDescriptionResult.GetDescription(),
                                                 profiles,
                                                 databases,
                                                 postprocessors,
                                                 motorwayTypeNames,
                                                 motorwayLinkTypeNames,
                                                 junctionTypeNames)) {
    std::cerr << "Error during route postprocessing" << std::endl;
  }

  postprocessTimer.Stop();

  std::cout << "Postprocessing time: " << postprocessTimer.ResultString() << std::endl;

  osmscout::StopClock                     generateTimer;
  osmscout::RouteDescriptionPostprocessor generator;
  // RouteDescriptionGeneratorCallback       generatorCallback(args.routeDebug);

  // generator.GenerateDescription(*routeDescriptionResult.GetDescription(),
  //                               generatorCallback);


  generateTimer.Stop();

  std::cout << "Description generation time: " << generateTimer.ResultString() << std::endl;

  router->Close();

  return 0;
}