#!/bin/bash
for nS in 250 350 450 550
do
  for nD in 100
  do
    for pm in reset
    do
      sbatch ./evaluate_models_round6_sbatch.sh $nS $nD $pm
    done
  done
done
