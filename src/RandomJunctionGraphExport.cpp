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
#include <random>

#include <osmscout/db/Database.h>

#include <osmscout/routing/SimpleRoutingService.h>
#include <osmscout/routing/RoutePostprocessor.h>
#include <osmscout/routing/DBFileOffset.h>
#include <osmscout/routing/RouteDescriptionPostprocessor.h>

#include <osmscout/cli/CmdLineParsing.h>
#include <osmscout/util/Bearing.h>
#include <osmscout/util/Geometry.h>

#include <JunctionGraphProcessor.h>

struct Arguments
{
  bool                              help=false;
  std::string                       router=osmscout::RoutingService::DEFAULT_FILENAME_BASE;
  osmscout::Vehicle                 vehicle=osmscout::Vehicle::vehicleCar;
  std::string                       databaseDirectory;
  std::filesystem::path             junctionExportDir=std::filesystem::current_path();
  int                               numberOfRoutes=100;
  bool                              debug=false;

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

// Function to get random routing nodes from the database bounds
std::tuple<osmscout::RoutePosition, osmscout::RoutePosition> GetRandomCoordinates(
    const osmscout::DatabaseRef& database,
    const std::vector<osmscout::RoutePosition> &routableWays,
    std::mt19937& rng)
{
  return {routableWays[rng() % routableWays.size()],
          routableWays[rng() % routableWays.size()]};
}

bool ProcessRoute(osmscout::SimpleRoutingServiceRef& router,
                  osmscout::FastestPathRoutingProfileRef& routingProfile,
                  osmscout::DatabaseRef& database,
                  const osmscout::RoutePosition& start,
                  const osmscout::RoutePosition& target,
                  const std::filesystem::path& junctionExportDir,
                  int routeIndex)
{
  using namespace osmscout;

  RoutePosition startPos = start;
  RoutePosition targetPos = target;

  RoutingParameter parameter;
  parameter.SetProgress(std::make_shared<ConsoleRoutingProgress>());

  RoutingResult result = router->CalculateRoute(*routingProfile, startPos, targetPos, {}, parameter);

  if (!result.Success()) {
    std::cerr << "Cannot calculate route " << routeIndex << std::endl;
    return false;
  }

  auto routeDescriptionResult = router->TransformRouteDataToRouteDescription(result.GetRoute());
  if (!routeDescriptionResult.Success()) {
    std::cerr << "Error during generation of route description for route " << routeIndex << std::endl;
    return false;
  }

  std::list<RoutePostprocessor::PostprocessorRef> postprocessors{
    std::make_shared<RoutePostprocessor::DistanceAndTimePostprocessor>(),
    std::make_shared<RoutePostprocessor::StartPostprocessor>("Start"),
    std::make_shared<RoutePostprocessor::TargetPostprocessor>("Target"),
    std::make_shared<RoutePostprocessor::WayNamePostprocessor>(),
    std::make_shared<RoutePostprocessor::WayTypePostprocessor>(),
    std::make_shared<RoutePostprocessor::CrossingWaysPostprocessor>(),
    std::make_shared<RoutePostprocessor::DirectionPostprocessor>(),
    std::make_shared<RoutePostprocessor::LanesPostprocessor>(),
    std::make_shared<RoutePostprocessor::SuggestedLanesPostprocessor>(),
    std::make_shared<RoutePostprocessor::MotorwayJunctionPostprocessor>(),
    std::make_shared<RoutePostprocessor::DestinationPostprocessor>(),
    std::make_shared<RoutePostprocessor::MaxSpeedPostprocessor>(),
    std::make_shared<RoutePostprocessor::InstructionPostprocessor>(),
    std::make_shared<RoutePostprocessor::POIsPostprocessor>(),
    std::make_shared<JunctionGraphProcessor>(junctionExportDir),
  };

  RoutePostprocessor postprocessor;

  std::set<std::string,std::less<>> motorwayTypeNames{"highway_motorway",
                                                      "highway_motorway_trunk",
                                                      "highway_trunk",
                                                      "highway_motorway_primary"};
  std::set<std::string,std::less<>> motorwayLinkTypeNames{"highway_motorway_link",
                                                          "highway_trunk_link"};
  std::set<std::string,std::less<>> junctionTypeNames{"highway_motorway_junction"};

  std::vector<RoutingProfileRef> profiles{routingProfile};
  std::vector<DatabaseRef> databases{database};

  postprocessors.push_back(std::make_shared<RoutePostprocessor::SectionsPostprocessor>(result.GetSectionLenghts()));

  if (!postprocessor.PostprocessRouteDescription(*routeDescriptionResult.GetDescription(),
                                                 profiles,
                                                 databases,
                                                 postprocessors,
                                                 motorwayTypeNames,
                                                 motorwayLinkTypeNames,
                                                 junctionTypeNames)) {
    std::cerr << "Error during route postprocessing for route " << routeIndex << std::endl;
    return false;
  }

  std::cout << "Route " << routeIndex << " processed successfully" << std::endl;
  return true;
}

int main(int argc, char* argv[]) {
  using namespace osmscout;
  using namespace std::string_literals;
  using namespace std::chrono;

  CmdLineParser argParser("Random Junction Graph Export",
                          argc, argv);
  std::vector<std::string> helpArgs{"h","help"};
  Arguments args;

  argParser.AddOption(CmdLineFlag([&args](const bool& value) {
                        args.help=value;
                      }),
                      helpArgs,
                      "Return argument help",
                      true);

  argParser.AddOption(CmdLineStringOption([&args](const std::string& value) {
                        args.junctionExportDir=value;
                      }),
                      "junctionExportDir",
                      "Directory for exporting junction JSON files",
                      false);

  argParser.AddOption(CmdLineFlag([&args](const bool& value) {
                        args.debug=value;
                      }),
                      "debug",
                      "Enable debug output",
                      false);

  argParser.AddOption(CmdLineAlternativeFlag([&args](const std::string& value) {
                        if (value=="foot") {
                          args.vehicle=Vehicle::vehicleFoot;
                        }
                        else if (value=="bicycle") {
                          args.vehicle=Vehicle::vehicleBicycle;
                        }
                        else if (value=="car") {
                          args.vehicle=Vehicle::vehicleCar;
                        }
                      }),
                      {"foot","bicycle","car"},
                      "Vehicle type to use for routing");

  argParser.AddOption(CmdLineStringOption([&args](const std::string& value) {
                        args.router=value;
                      }),
                      "router",
                      "Router filename base");

  argParser.AddOption(CmdLineUIntOption([&args](unsigned int value) {
                        args.numberOfRoutes=value;
                      }),
                      "numberOfRoutes",
                      "Number of random routes to generate");


  argParser.AddPositional(CmdLineStringOption([&args](const std::string& value) {
                            args.databaseDirectory=value;
                          }),
                          "DATABASE",
                          "Directory of the db to use");


  CmdLineParseResult cmdLineParseResult=argParser.Parse();

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

  // Initialize random number generator
  std::random_device rd;
  std::mt19937 rng(rd());

  DatabaseParameter databaseParameter;
  DatabaseRef database = std::make_shared<Database>(databaseParameter);

  if (!database->Open(args.databaseDirectory)) {
    std::cerr << "Cannot open database: " << args.databaseDirectory << std::endl;
    return 1;
  }

  FastestPathRoutingProfileRef routingProfile = std::make_shared<FastestPathRoutingProfile>(database->GetTypeConfig());
  RouterParameter routerParameter;

  routingProfile->SetPenaltySameType(args.penaltySameType);
  routingProfile->SetPenaltyDifferentType(args.penaltyDifferentType);
  routingProfile->SetMaxPenalty(args.maxPenalty);

  routerParameter.SetDebugPerformance(true);

  SimpleRoutingServiceRef router = std::make_shared<SimpleRoutingService>(database,
                                                                          routerParameter,
                                                                          args.router);

  if (!router->Open()) {
    std::cerr << "Cannot open routing db" << std::endl;
    return 1;
  }

  TypeConfigRef typeConfig = database->GetTypeConfig();
  std::map<std::string,double> carSpeedTable;

  switch (args.vehicle) {
    case vehicleFoot:
      routingProfile->ParametrizeForFoot(*typeConfig, 5.0);
      break;
    case vehicleBicycle:
      routingProfile->ParametrizeForBicycle(*typeConfig, 20.0);
      break;
    case vehicleCar:
      GetCarSpeedTable(carSpeedTable);
      routingProfile->ParametrizeForCar(*typeConfig, carSpeedTable, 160.0);
      break;
  }

  // there is no high-level api for reading all the ways, we need to use low-level...
  std::cout << "Reading routable ways from database..." << std::endl;
  FileScanner wayScanner;
  std::vector<RoutePosition> routableWays;
  try {
    wayScanner.Open(database->GetWayDataFile()->GetFilename(), FileScanner::Sequential, false);
    uint32_t wayCount=wayScanner.ReadUInt32();

    // Iterate through each way and add text
    // data to the corresponding keyset
    for(uint32_t n=1; n <= wayCount; n++) {
      Way way;
      way.Read(*typeConfig, wayScanner);
      if (routingProfile->CanUse(way) && std::any_of(way.nodes.begin(), way.nodes.end(),
                                                     [](const Point& node) {
                                                       return node.IsRelevant();
                                                     })) {
        routableWays.emplace_back(way.GetObjectFileRef(), /*nodeIndex*/0, /*db*/0);
      }
    }
    wayScanner.Close();
  } catch (IOException& e) {
    osmscout::log.Error() << "Failed to read ways: " << e.GetDescription();
    return 1;
  }
  std::cout << "Found " << routableWays.size() << " routable ways." << std::endl;

  // Create export directory if it doesn't exist
  std::filesystem::create_directories(args.junctionExportDir);

  std::cout << "Generating " << args.numberOfRoutes << " random routes..." << std::endl;

  int successfulRoutes = 0;
  int attempts = 0;
  const int maxAttempts = args.numberOfRoutes * 3; // Allow some failed attempts

  while (successfulRoutes < args.numberOfRoutes && attempts < maxAttempts) {
    attempts++;

    // Generate random coordinates within database bounds
    auto [from, to] = GetRandomCoordinates(database, routableWays, rng);

    if (ProcessRoute(router, routingProfile, database, from, to,
                     args.junctionExportDir, successfulRoutes + 1)) {
      successfulRoutes++;
    }
  }

  std::cout << "Successfully processed " << successfulRoutes << " out of "
            << args.numberOfRoutes << " requested routes (" << attempts << " attempts)" << std::endl;

  router->Close();

  return 0;
}
