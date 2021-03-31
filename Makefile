CXX = mpicxx

USE_TAUPROF=0
ifeq ($(USE_TAUPROF),1)
TAU=/soft/perftools/tau/tau-2.29/craycnl/lib
CXX = tau_cxx.sh -tau_makefile=$(TAU)/Makefile.tau-intel-papi-mpi-pdt 
endif

# use -xmic-avx512 instead of -xHost for Intel Xeon Phi platforms
OPTFLAGS = -O3 -fopenmp -DPRINT_DIST_STATS 
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
SNTFLAGS = -std=c++11 -fopenmp -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++11 -g $(OPTFLAGS)

ENABLE_DUMPI_TRACE=0
ENABLE_SCOREP_TRACE=0
ifeq ($(ENABLE_DUMPI_TRACE),1)
	TRACERPATH = $(HOME)/builds/sst-dumpi/lib 
	LDFLAGS = -L$(TRACERPATH) -ldumpi
else ifeq ($(ENABLE_SCOREP_TRACE),1)
	SCOREP_INSTALL_PATH = /usr/common/software/scorep/6.0/intel
	INCLUDE = -I$(SCOREP_INSTALL_PATH)/include -I$(SCOREP_INSTALL_PATH)/include/scorep -DSCOREP_USER_ENABLE
	LDAPP = $(SCOREP_INSTALL_PATH)/bin/scorep --user --nocompiler --noopenmp --nopomp --nocuda --noopenacc --noopencl --nomemory
endif

OBJ = main.o
TARGET = nido 

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $^

$(TARGET):  $(OBJ)
	$(CXX) $^ $(OPTFLAGS) -o $@

.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET)
