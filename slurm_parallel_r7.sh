#!/bin/bash
for nS in 10 15
do
  for nD in 100
  do
    for pm in reset
    do
      for lp in 0.0025 0.005 0.0075 0.01
      do	      
        sbatch ./evaluate_models_round7_sbatch.sh $nS $nD $pm $lp
      done	
    done
  done
done
