#!/usr/bin/env bash

datafile='CDs'

#keeps=('0.6' '0.8' '0.9')
keeps=('0.8' '0.9' '0.7' '0.6' '0.5')

seeds=('2017' '2018' '2019' '2020' '2021')
#seeds=('2017')

for keep in ${keeps[@]};do
for seed in ${seeds[@]};do
CUDA_VISIBLE_DEVICES="0" python2 -u train.py $datafile $keep $seed > $datafile'_'$keep'_'$seed
done
done