MPI_HOME ?= /usr/local/mpi
MPI_INCLUDE = $(MPI_HOME)/include
MPI_LIB = $(MPI_HOME)/lib
MPI_LINK = -lmpi
COMPILER = g++
EXTRA_COMPILER_FLAG = -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wl,-rpath,../../build/lib -std=c++11 -g

INCLUDEDIR := $(abspath ../perf/include)
LIBSRCFILES:= $(wildcard *.cc)

all: test-kernel-demo

test-kernel-demo: test_kernel_demo.cc
  @echo "Compiling $@"
  @$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_kernel_demo test_kernel_demo.cc -I../../flagcx/include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -L../../build/lib -L$(MPI_LIB) -lflagcx $(MPI_LINK)

clean:
  @rm -f test_kernel_demo

print_var:
	@echo "INCLUDEDIR: $(INCLUDEDIR)"
  @echo "USE_NVIDIA: $(USE_NVIDIA)"
  @echo "MPI_INCLUDE: $(MPI_INCLUDE)"
  @echo "MPI_LIB: $(MPI_LIB)"
  @echo "COMPILER: $(COMPILER)"
  @echo "EXTRA_COMPILER_FLAG: $(EXTRA_COMPILER_FLAG)"