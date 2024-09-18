cmake_minimum_required(VERSION 3.15)
project(agent VERSION 1.0 DESCRIPTION "Model agent" LANGUAGES CXX)

# Set C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -g -O2")
add_compile_options(-Wall -Wextra -pedantic)

# Find dependencies
find_package(Torch REQUIRED)
find_package(drogon REQUIRED)

# Define the source files for the executable
set(SOURCE_FILES
    src/Ai/Model.cpp
    src/Ai/Simulate.cpp
    src/Ai/Score.cpp
    src/Core/6502.cpp
    src/Core/Controller.cpp
    src/Core/PPU.cpp
    src/Core/RAM.cpp
    src/Core/ROM.cpp
    src/Core/Mapper/CNROM.cpp
    src/Core/Mapper/MMC1.cpp
    src/Core/Mapper/Mapper.cpp
    src/Core/Mapper/NROM.cpp
    src/Core/Mapper/UnROM.cpp
    src/Web/Luigi.cpp
    src/Web/Episode.cpp
    # Uncomment the line below if you want to include the main.cpp file
    # Source/Desktop/Main.cpp
)

set(MEDNES_FILES
    src/Core/6502.cpp
    src/Core/Controller.cpp
    src/Core/PPU.cpp
    src/Core/RAM.cpp
    src/Core/ROM.cpp
    src/Core/Mapper/CNROM.cpp
    src/Core/Mapper/MMC1.cpp
    src/Core/Mapper/Mapper.cpp
    src/Core/Mapper/NROM.cpp
    src/Core/Mapper/UnROM.cpp
)

set_source_files_properties(
    ${MEDNES_FILES}
    PROPERTIES
    COMPILE_FLAGS "-Wno-everything"
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE Drogon::Drogon)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add a custom target for cleaning more thoroughly
add_custom_target(clean-all
   COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}.dir/cmake_clean.cmake
)