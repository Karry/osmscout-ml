#pragma once


#include <osmscout/routing/RoutePostprocessor.h>

namespace osmscout {

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

};

}
