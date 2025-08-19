cmake_minimum_required(VERSION 3.16)
project(OSMScoutML)
find_package(libosmscout REQUIRED COMPONENTS OSMScout Map Client)


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
