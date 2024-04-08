project('MedNES', 'cpp', default_options: ['cpp_std=c++14', 'buildtype=release', 'warning_level=3'])

cpp = meson.get_compiler('cpp')

# Assuming dependencies are handled here
libtorch_include = include_directories('/libtorch/include', '/libtorch/include/torch/csrc/api/include')
libtorch_lib = cpp.find_library('torch', dirs: ['/libtorch/lib'])

# Manually list all source files
src_files = files(
    'Source/Ai/Model.cpp',
    'Source/Ai/Simulate.cpp',
    'Source/Core/6502.cpp',
    'Source/Core/Controller.cpp',
    'Source/Core/INESBus.hpp',
    'Source/Core/PPU.cpp',
    'Source/Core/RAM.cpp',
    'Source/Core/ROM.cpp',
    'Source/Core/Common/Typedefs.hpp',
    'Source/Core/Mapper/CNROM.cpp',
    'Source/Core/Mapper/MMC1.cpp',
    'Source/Core/Mapper/Mapper.cpp',
    'Source/Core/Mapper/NROM.cpp',
    'Source/Core/Mapper/UnROM.cpp'
    #'Source/Desktop/Main.cpp'
)

# Specify include directories if needed
include_dirs = include_directories('Source', 'Source/Core', 'Source/Core/Common', 'Source/Core/Mapper', 'Source/Ai')

# Create the executable
executable('MedNES', src_files, include_directories: include_dirs, dependencies: [libtorch_lib])
