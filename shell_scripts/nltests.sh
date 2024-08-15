#!/bin/bash

# List of values to iterate over
lvalues=(14 26 38 50 74 86 110 146 170 194 230 266 302 350 434 590 770 974)
rvalues=(10 16 56 63 69 74 79 84 89 97 105)

# Log file where the output will be appended
logfile="output_nl.log"

# Clear the logfile if it exists (optional)
# > "$logfile"

# Loop through each value and execute the command
for i in "${rvalues[@]}"; do
  for j in "${lvalues[@]}"; do
    echo "Running: ./simple -r $i -l $j"
    ./simple -r "$i" -l "$j" >> "$logfile"
  done
done
