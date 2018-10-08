CXX ?= g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++11 -fopenmp

.PHONY: all clean run plot

all: alock

clean:
	rm -f alock results.txt results-rearranged.txt results.png

run:
	./alock | tee results.txt
	# Results stored into 'results.txt', run 'make plot'.

plot:
	python3 plot.py

alock: alock.cpp Makefile
	$(CXX) $(CXXFLAGS) alock.cpp -o alock

