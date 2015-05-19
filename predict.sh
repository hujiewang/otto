#!/bin/bash
FILE="./results/results.csv"
for i in `seq 90022 90039`;
do
   th predict.lua -file "./model/$i.dat" -valid -randFeat
   mv "${FILE}" "./python/valid_results/valid_$i.csv"
   th predict.lua -file "./model/$i.dat" -randFeat
   mv "${FILE}" "./python/results/results_$i.csv"
done   
