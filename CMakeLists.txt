cmake_minimum_required(VERSION 3.16)
project(OSMScoutML)
find_package(libosmscout REQUIRED COMPONENTS OSMScout Map)

set(SOURCE_FILES
    src/JunctionGraphExport.cpp
    src/JunctionGraphProcessor.cpp
    src/JunctionGraphProcessor.h
)
add_executable(JunctionGraphExport ${SOURCE_FILES})
target_include_directories(JunctionGraphExport PRIVATE src)

set_property(TARGET JunctionGraphExport PROPERTY CXX_STANDARD 20)
target_link_libraries(JunctionGraphExport PRIVATE OSMScout::OSMScout)
