CXX ?= g++
CXXFLAGS = -std=c++11 -Wall -Wextra -Wpedantic

power_method: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lopenblas

measurement: power_method plot.py
	./$< > results.txt; python plot.py

.PHONY: clean
clean:
	@rm -rf *.png power_method results.txt