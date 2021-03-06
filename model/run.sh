#!/usr/bin/env bash

datafile='Movies'

#keeps=('0.6' '0.8' '0.9')
keeps=('1.0')

seeds=('2017' '2018' '2019' '2020' '2021')
seeds=('2019' '2020' '2021')

for keep in ${keeps[@]};do
for seed in ${seeds[@]};do
CUDA_VISIBLE_DEVICES="3" python2 -u train.py $datafile $keep $seed > $datafile'_'$keep'_'$seed
done
done