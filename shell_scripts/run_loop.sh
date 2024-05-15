#!/bin/bash

# Define the starting value
start=0.01

# Define the number of iterations
iterations=12

# Loop over the iterations
for ((i = 0; i < iterations; i++)); do
    # Run the program with the current value
    ./simple t $start

    # Update the value for the next iteration (reduce by one order of magnitude)
    start=$(bc <<< "scale=10; $start / 10")    
done

