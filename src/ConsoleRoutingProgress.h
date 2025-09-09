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

#include <osmscout/routing/SimpleRoutingService.h>
#include <chrono>

#pragma once


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
