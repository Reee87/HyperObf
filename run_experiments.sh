#!/bin/bash

echo "Starting experiments..."

# Define parameter values
threshold_values=(0.0001 0.001 0.01)
diversity_lamba_values=(1 10 100)

# Run experiments for all parameter combinations
for threshold in "${threshold_values[@]}"; do
    for diversity_lamba in "${diversity_lamba_values[@]}"; do
        echo "Running experiment with threshold=$threshold and diversity_lamba=$diversity_lamba"
        # Run training script with current parameter combination
        python3 train_hypernet.py --lr 0.0001 --wd 0 --cuda --batch_size 8 --epochs 850 --threshold "$threshold" --diversity_lambda "$diversity_lamba"
        echo "Experiment completed."
    done
done

echo "All experiments completed."
