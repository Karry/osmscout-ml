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


set(JUNCTION_EXPORT_SOURCE_FILES
    src/JunctionGraphExport.cpp
    src/JunctionGraphProcessor.cpp
    src/JunctionGraphProcessor.h
)
add_executable(JunctionGraphExport ${JUNCTION_EXPORT_SOURCE_FILES})
target_include_directories(JunctionGraphExport PRIVATE src)
set_property(TARGET JunctionGraphExport PROPERTY CXX_STANDARD 20)
target_link_libraries(JunctionGraphExport PRIVATE OSMScout::OSMScout)


set(RANDOM_EXPORT_SOURCE_FILES
    src/RandomJunctionGraphExport.cpp
    src/JunctionGraphProcessor.cpp
    src/JunctionGraphProcessor.h
)
add_executable(RandomJunctionGraphExport ${RANDOM_EXPORT_SOURCE_FILES})
target_include_directories(RandomJunctionGraphExport PRIVATE src)
set_property(TARGET RandomJunctionGraphExport PROPERTY CXX_STANDARD 20)
target_link_libraries(RandomJunctionGraphExport PRIVATE OSMScout::OSMScout)

if(Torch_FOUND)
    set(PREDICT_JUNCTION_SOURCE_FILES
        src/PredictJunction.cpp
        src/JunctionGraphProcessor.cpp
        src/JunctionGraphProcessor.h
    )
    add_executable(PredictJunction ${PREDICT_JUNCTION_SOURCE_FILES})
    target_include_directories(PredictJunction PRIVATE src)
    set_property(TARGET PredictJunction PROPERTY CXX_STANDARD 20)
    target_link_libraries(PredictJunction PRIVATE OSMScout::OSMScout ${TORCH_LIBRARIES})
endif()
