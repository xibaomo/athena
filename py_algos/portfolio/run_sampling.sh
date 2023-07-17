#!/bin/bash

if [[ $# < 2 ]];then
    echo "Usage: ./run_sampling <date> <num_iterations>"
    exit 1
fi

for ((i=1; i<=$2; i++))
do
    echo "################### $i-th run #####################"
    python max_std.py port.yaml $1
done
