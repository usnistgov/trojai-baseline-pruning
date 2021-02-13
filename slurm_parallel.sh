#!/bin/bash
for nS in 15 25 35
do
  for nD in 10 500
  do
    for pm in reset remove
    do
      sbatch ./evaluate_models_round3_sbatch.sh $nS $nD $pm
    done
  done
done
