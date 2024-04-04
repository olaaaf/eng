project('MedNES', 'cpp', default_options : ['cpp_std=c++14', 'buildtype=release', 'warning_level=3'])

# Dependencies
sdl2_dep = dependency('sdl2')
pytorch_dep = cpp.find_library('torch', required : true)

# Sources
src_files = files(
    'Source/Core/*.cpp',
    'Source/Core/Mapper/*.cpp',
    'Source/Core/Common/*.cpp',
    'Source/Ai/*.cpp'
)

# Executable
executable('MedNES', src_files, dependencies : [sdl2_dep, pytorch_dep])

# Clean target is implicit in Meson through 'meson compile --clean' or 'ninja -C build clean'
