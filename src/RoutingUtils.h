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

#include <map>
#include <string>
#include <list>
#include <chrono>
#include <iostream>
#include <optional>
#include <filesystem>

#include <osmscout/db/Database.h>

#include <osmscout/routing/SimpleRoutingService.h>
#include <osmscout/routing/RoutePostprocessor.h>
#include <osmscout/routing/RouteDescriptionPostprocessor.h>

#include <osmscout/util/Bearing.h>

#include <ConsoleRoutingProgress.h>
#include <RoutingUtils.h>


extern void GetCarSpeedTable(std::map<std::string,double>& map);

inline int ComputeRoute(const std::string &databaseDirectory,
                        const std::list<osmscout::RoutePostprocessor::PostprocessorRef> &postprocessors,
                        osmscout::GeoCoord startCoord,
                        osmscout::GeoCoord targetCoord,
                        std::vector<osmscout::GeoCoord> via={},
                        const osmscout::Distance penaltySameType=osmscout::Meters(40),
                        const osmscout::Distance penaltyDifferentType=osmscout::Meters(250),
                        const osmscout::HourDuration maxPenalty=std::chrono::seconds(10),
                        std::string routerFileBase=osmscout::RoutingService::DEFAULT_FILENAME_BASE,
                        osmscout::Vehicle vehicle=osmscout::Vehicle::vehicleCar,
                        std::optional<osmscout::Bearing>  initialBearing=std::nullopt,
                        bool dataDebug=false
                        )
{
  osmscout::DatabaseParameter databaseParameter;
  osmscout::DatabaseRef       database=std::make_shared<osmscout::Database>(databaseParameter);

  if (!database->Open(databaseDirectory)) {
    std::cerr << "Cannot open db" << std::endl;
    return 1;
  }

  osmscout::FastestPathRoutingProfileRef routingProfile=std::make_shared<osmscout::FastestPathRoutingProfile>(database->GetTypeConfig());
  osmscout::RouterParameter              routerParameter;

  routingProfile->SetPenaltySameType(penaltySameType);
  routingProfile->SetPenaltyDifferentType(penaltyDifferentType);
  routingProfile->SetMaxPenalty(maxPenalty);

  routerParameter.SetDebugPerformance(true);

  osmscout::SimpleRoutingServiceRef router=std::make_shared<osmscout::SimpleRoutingService>(database,
                                                                                            routerParameter,
                                                                                            routerFileBase);

  if (!router->Open()) {
    std::cerr << "Cannot open routing db" << std::endl;

    return 1;
  }

  osmscout::TypeConfigRef             typeConfig=database->GetTypeConfig();
  std::map<std::string,double>        carSpeedTable;
  osmscout::RoutingParameter          parameter;

  parameter.SetProgress(std::make_shared<ConsoleRoutingProgress>());

  switch (vehicle) {
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

  auto startResult=router->GetClosestRoutableNode(startCoord,
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

  auto targetResult=router->GetClosestRoutableNode(targetCoord,
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

  if (via.size() > 0) {
    std::cout << "Using 'CalculateRouteViaCoords' method" << std::endl;
    via.insert(via.begin(), startCoord);
    via.push_back(targetCoord);
    result=router->CalculateRouteViaCoords(*routingProfile,
                                           via,
                                           osmscout::Kilometers(1),
                                           parameter);

  } else {
    std::cout << "Using 'CalculateRoute' method" << std::endl;
    result=router->CalculateRoute(*routingProfile,
                                  start,
                                  target,
                                  initialBearing,
                                  parameter);
  }

  if (!result.Success()) {
    std::cerr << "There was an error while calculating the route!" << std::endl;
    router->Close();
    return 1;
  }

  if (dataDebug) {
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