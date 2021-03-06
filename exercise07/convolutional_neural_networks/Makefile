config ?= debug

ifeq ($(shell uname -s), Darwin)
CXX=g++-8
LIBS += -L/usr/local/opt/openblas/lib/ -lopenblas
CXXFLAGS += -I/usr/local/opt/openblas/include/
else
CXX=g++ -lopenblas
endif

CXXFLAGS+= -std=c++14 -fopenmp

ifeq "$(config)" "debug"
CXXFLAGS += -g -O0
# addressing errors generally involve accessing/writing beyond the bounds of
# the allocated memory space (i.e. seg fault)
#CXXFLAGS += -fsanitize=address
# undefined behaviors range from operation outcomes depending on uninitialized
# data to wron memory alignment:
#CXXFLAGS += -fsanitize=undefined
endif

ifeq "$(config)" "prod"
CXXFLAGS += -DNDEBUG -O3 -ffast-math
endif


CXXFLAGS+= -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wpedantic
CXXFLAGS+= -Wmissing-declarations -Wredundant-decls -Wshadow -Wwrite-strings
CXXFLAGS+= -Woverloaded-virtual -Wno-unused-parameter -Wno-unused-variable


exec_testGrad: main_testGrad.o
	$(CXX) $(CXXFLAGS) $(LIBS) main_testGrad.o -o $@ -lopenblas
	rm main_testGrad.o

exec_classify: main_classify.o
	$(CXX) $(CXXFLAGS) $(LIBS) main_classify.o -o $@ -lopenblas
	rm main_classify.o

exec_classify_improved: main_classify_improved.o
	$(CXX) $(CXXFLAGS) $(LIBS) main_classify_improved.o -o $@ -lopenblas
	rm main_classify_improved.o

all: exec_testGrad exec_classify exec_classify_improved
.DEFAULT_GOAL := all

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.dSYM *.s *.d exec_*
