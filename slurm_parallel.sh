#!/bin/bash
for nS in 15
do
  for nD in 100
  do
    for pm in reset
    do
      sbatch ./evaluate_models_round7_sbatch.sh $nS $nD $pm
    done
  done
done
