#!/bin/bash

# Check for missing configurations and write to missing_conf.txt
> missing_conf.txt
for i in {0..49}; do
    if [ ! -f "/home/jinchen/git/lat-software/gpt_benchmark/conf/S16T16_cg/gauge/wilson_b6.cg.1e-08.$i" ]; then
        echo $i >> missing_conf.txt
    fi
done

# Run gauge fixing for each missing configuration
while read n_conf; do
    echo "Running gauge fixing for configuration $n_conf"
    python gfix_cgpt.py $n_conf
done < missing_conf.txt
