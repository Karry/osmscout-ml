cmake_minimum_required(VERSION 3.16)
project(OSMScoutML)
find_package(libosmscout REQUIRED COMPONENTS OSMScout Map Client)

# Find PyTorch++
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "Found PyTorch: ${TORCH_LIBRARIES}")
    set(TORCH_FOUND TRUE)
else()
    message(WARNING "PyTorch not found. PredictJunction will not be built.")
    set(TORCH_FOUND FALSE)
endif()

set(ROUTING_UTILS_SOURCE_FILES
    src/RoutingUtils.cpp
    src/RoutingUtils.h
    src/JunctionGraphProcessor.cpp
    src/JunctionGraphProcessor.h
)
add_library(RoutingUtils ${ROUTING_UTILS_SOURCE_FILES})
target_include_directories(RoutingUtils PRIVATE src)
set_property(TARGET RoutingUtils PROPERTY CXX_STANDARD 20)
target_link_libraries(RoutingUtils PRIVATE OSMScout::OSMScout)


set(JUNCTION_EXPORT_SOURCE_FILES
    src/JunctionGraphExport.cpp
)
add_executable(JunctionGraphExport ${JUNCTION_EXPORT_SOURCE_FILES})
target_include_directories(JunctionGraphExport PRIVATE src)
set_property(TARGET JunctionGraphExport PROPERTY CXX_STANDARD 20)
target_link_libraries(JunctionGraphExport PRIVATE OSMScout::OSMScout RoutingUtils)


set(RANDOM_EXPORT_SOURCE_FILES
    src/RandomJunctionGraphExport.cpp
)
add_executable(RandomJunctionGraphExport ${RANDOM_EXPORT_SOURCE_FILES})
target_include_directories(RandomJunctionGraphExport PRIVATE src)
set_property(TARGET RandomJunctionGraphExport PROPERTY CXX_STANDARD 20)
target_link_libraries(RandomJunctionGraphExport PRIVATE OSMScout::OSMScout RoutingUtils)

if(Torch_FOUND)
    set(PREDICT_JUNCTION_SOURCE_FILES
        src/PredictJunction.cpp
    )
    add_executable(PredictJunction ${PREDICT_JUNCTION_SOURCE_FILES})
    target_include_directories(PredictJunction PRIVATE src)
    set_property(TARGET PredictJunction PROPERTY CXX_STANDARD 20)
    target_link_libraries(PredictJunction PRIVATE OSMScout::OSMScout ${TORCH_LIBRARIES} RoutingUtils)

    set(EVALUATE_JUNCTIONS_SOURCE_FILES
        src/EvaluateJunctions.cpp
    )
    add_executable(EvaluateJunctions ${EVALUATE_JUNCTIONS_SOURCE_FILES})
    target_include_directories(EvaluateJunctions PRIVATE src)
    set_property(TARGET EvaluateJunctions PROPERTY CXX_STANDARD 20)
    target_link_libraries(EvaluateJunctions PRIVATE OSMScout::OSMScout ${TORCH_LIBRARIES} RoutingUtils)
endif()

# DrawJunction - junction map visualization tool (requires Qt)
find_package(Qt5 COMPONENTS Gui QUIET)
if(Qt5_FOUND)
    add_executable(DrawJunction src/DrawJunction.cpp)
    target_include_directories(DrawJunction PRIVATE src)
    set_property(TARGET DrawJunction PROPERTY CXX_STANDARD 20)
    target_link_libraries(DrawJunction PRIVATE
        OSMScout::OSMScout OSMScout::Map OSMScout::MapQt
        RoutingUtils Qt5::Gui)
    if(Torch_FOUND)
        target_link_libraries(DrawJunction PRIVATE ${TORCH_LIBRARIES})
        target_compile_definitions(DrawJunction PRIVATE HAS_TORCH=1)
    endif()
else()
    message(WARNING "Qt5 not found. DrawJunction will not be built.")
endif()
