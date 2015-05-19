#!/bin/bash
FILE="./model/model.dat"
for i in `seq 90025 91000`;
do
   th main.lua
   mv "${FILE}" "./model/$i.dat"
done   
