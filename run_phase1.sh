#!/bin/bash

SEED=42
NUM_CHAINS=1
DATASET=synthetic
SAMPLES_LIST=(1000 2000)
K_LIST=(4 10 20)
PERCW_LIST=(25 50)

# Calculate total number of runs (2 for noise/no noise)
TOTAL_RUNS=$(( ${#K_LIST[@]} * ${#PERCW_LIST[@]} * ${#SAMPLES_LIST[@]} * 2 ))
RUN_COUNT=1

for K in "${K_LIST[@]}"; do
  for PERCW in "${PERCW_LIST[@]}"; do
    for SAMPLES in "${SAMPLES_LIST[@]}"; do
      
      # Without noise
      echo "Run $RUN_COUNT of $TOTAL_RUNS: K=$K percW=$PERCW samples=$SAMPLES NO NOISE"
      python run_analysis.py \
        --dataset $DATASET \
        --K $K \
        --percW $PERCW \
        --num-samples $SAMPLES \
        --num-chains $NUM_CHAINS \
        --seed $SEED
      ((RUN_COUNT++))
      
      # With noise
      echo "Run $RUN_COUNT of $TOTAL_RUNS: K=$K percW=$PERCW samples=$SAMPLES WITH NOISE"
      python run_analysis.py \
        --dataset $DATASET \
        --K $K \
        --percW $PERCW \
        --num-samples $SAMPLES \
        --num-chains $NUM_CHAINS \
        --seed $SEED \
        --noise
      ((RUN_COUNT++))

    done
  done
done