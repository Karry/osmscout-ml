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

class JunctionGraphProcessor: public RoutePostprocessor::Postprocessor
{
private:
  std::filesystem::path exportDirectory;

public:
  JunctionGraphProcessor(const std::filesystem::path& exportDirectory);
  ~JunctionGraphProcessor() override = default;

  JunctionGraphProcessor(const JunctionGraphProcessor&) = delete;
  JunctionGraphProcessor& operator=(const JunctionGraphProcessor&) = delete;

  JunctionGraphProcessor(JunctionGraphProcessor&&) = delete;
  JunctionGraphProcessor& operator=(JunctionGraphProcessor&&) = delete;

  bool Process(const PostprocessorContext& context,
               RouteDescription& description) override;

};

using JunctionGraphProcessorRef = std::shared_ptr<JunctionGraphProcessor>;
}
