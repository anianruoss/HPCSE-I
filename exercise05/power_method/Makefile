CXX ?= g++
CXXFLAGS = -std=c++11 -Wall -Wextra -Wpedantic

power_method: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lopenblas

job: power_method
	bsub -n 24 -R fullnode ./$< -oo results.txt

plot: plot.py
	python plot.py

.PHONY: clean
clean:
	@rm -rf *.png power_method results.txt