#include <JunctionGraphProcessor.h>

namespace osmscout {

bool JunctionGraphProcessor::Process(const PostprocessorContext& context,
                                     RouteDescription& description) {

    for (auto& node : description.Nodes()) {
        if (node.HasDescription(RouteDescription::TURN_DESC) ||
                node.HasDescription(RouteDescription::MOTORWAY_CHANGE_DESC) ||
                node.HasDescription(RouteDescription::MOTORWAY_LEAVE_DESC) ||
                node.HasDescription(RouteDescription::MOTORWAY_JUNCTION_DESC)
            ) {
            log.Debug() << "Exporting junction graph for node "
                        << node.GetPathObject().GetFileOffset() << " / " << node.GetCurrentNodeIndex()
                        << " at " << node.GetLocation().GetDisplayText();
        }
    }
    return true;
};

}
