bin = MedNES
# Explicit list of source files
src = Source/Ai/Model.cpp \
      Source/Ai/Simulate.cpp \
	  Source/Ai/Utils.hpp \
      Source/Core/6502.cpp \
      Source/Core/Controller.cpp \
      Source/Core/INESBus.hpp \
      Source/Core/PPU.cpp \
      Source/Core/RAM.cpp \
      Source/Core/ROM.cpp \
      Source/Core/Common/Typedefs.hpp \
      Source/Core/Mapper/CNROM.cpp \
      Source/Core/Mapper/MMC1.cpp \
      Source/Core/Mapper/Mapper.cpp \
      Source/Core/Mapper/NROM.cpp \
      Source/Core/Mapper/UnROM.cpp 
#      Source/Desktop/Main.cpp

obj = $(src:.cpp=.o)

# Adjusted CXXFLAGS for libtorch paths based on the Dockerfile extraction path
CXXFLAGS = -g -Wall -Wextra -O2 -std=c++14 -pedantic \
           -I/libtorch/include \
           -I/libtorch/include/torch/csrc/api/include

# Assuming libtorch libraries are installed in a standard location inside the Docker container
LDFLAGS = -L/libtorch/lib -ltorch -lc10

.PHONY: all clean

all: $(bin)

$(bin): $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	-rm -f $(bin) $(obj)
