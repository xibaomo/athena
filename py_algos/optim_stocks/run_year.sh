#!/bin/bash

if [[ $# < 1 ]];then
    echo "Usage: ./run_year <year>"
    exit 1
fi

for ((i=1; i<=12; i++))
do
    date="$1-$i-01"
    echo "################### $date  #####################"
    python pick_opt.py port.yaml $date
done

