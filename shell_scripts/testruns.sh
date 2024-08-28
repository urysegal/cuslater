#!/bin/bash

# List of parameter values to iterate over
xvalues=(25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500)
# lvalues=()
# rvalues=()
# alphavalues=()

# Log file where outputs will be saved
logfile="tests/output_CHANGENAME.log"

# Clear the logfile if it exists (optional)
#> "$logfile"
echo "" >> "$logfile"

# Loop through values and execute the command

for x in "${xvalues[@]}"; do
  echo "Running: ./simple -n $x"
  ./simple -n "$x" >> "$logfile"
done