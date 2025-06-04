#!/bin/bash

precision=1e-08

# Check for missing configurations and write to missing_conf.txt
> missing_conf.txt
for i in {0..49}; do
    if [ ! -f "/home/jinchen/git/lat-software/gpt_benchmark/conf/S8T32_cg/gauge/wilson_b6.cg.${precision}.$i" ]; then
        echo $i >> missing_conf.txt
    fi
done

# Run gauge fixing for each missing configuration
while read n_conf; do
    echo "Running gauge fixing for configuration $n_conf with precision $precision"
    python gfix_cgpt.py $n_conf $precision
done < missing_conf.txt
