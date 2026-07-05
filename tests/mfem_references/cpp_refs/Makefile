# Makefile for MFEM cross-validation reference programs
# Build with MSYS2 mingw64

CXX = g++
CXXFLAGS = -std=c++17 -O2
INCLUDES = -I/mingw64/include -I/mingw64/include/suitesparse
LIBS = -L/mingw64/bin -lmfem -lumfpack -lamd -lcholmod -lsuitesparseconfig
LDFLAGS = -Wl,-rpath,/mingw64/bin

SRCS = ex31_ref.cpp ex32_ref.cpp ex33_ref.cpp ex34_ref.cpp ex22_ref.cpp
EXES = $(SRCS:.cpp=.exe)

all: $(EXES)

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS) $(LDFLAGS)

clean:
	rm -f $(EXES)

.PHONY: all clean
