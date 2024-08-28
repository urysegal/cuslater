#!/bin/bash

# List of parameter values to iterate over
# xvalues=()
# lvalues=()
# rvalues=()
# alphavalues=()

# Log file where outputs will be saved
logfile="output_CHANGENAME.log"

# Clear the logfile if it exists (optional)
#> "$logfile"
echo "" >> "$logfile"

# Loop through values and execute the command


for x in "${xvalues[@]}"; do
  echo "Running: ./simple -n $x"
  ./simple -n "$x" >> "$logfile"
done