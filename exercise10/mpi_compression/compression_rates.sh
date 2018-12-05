#!/usr/bin/env bash

rm compression_rates.txt compression_rates.png 2> /dev/null || true

for i in 0 1 2 4 8 16 32 64 128 256
do
    echo "# Tolerance $i" >> compression_rates.txt
    ./run.sh $i >> compression_rates.txt
done

python compression_rates.py
