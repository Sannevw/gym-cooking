#!/bin/bash

levels=("open-divider_tomato")

models=("greedy")

nagents=1
nseed=1

for seed in $(seq 1 1 $nseed); do
    for level in "${levels[@]}"; do
        for model1 in "${models[@]}"; do
            echo python main.py --num-agents $nagents --seed $seed --level $level --model1 $model1 --record
            python main.py --num-agents $nagents --seed $seed --level $level --model1 $model1 --record
            sleep 5
        done
    done
done
