#!/bin/bash
FILE="./model/model.dat"
for i in `seq 1 100`;
do
   th main.lua
   mv "${FILE}" "./model/$i.dat"
done   
