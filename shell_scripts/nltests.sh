#!/bin/bash

# List of values to iterate over
xyzvalues=(21 200 225 250 275 300 325 350 375 400 425 450 475 500 600 625 650 675 700 725 750)
lvalues=(14 26 38 50 74 86 110 146 170 194 230 266 302 350 434 590 770 974)
# rvalues=(2 10 16 56 63 69 74 79 84 89 97 105)
rvalues=(74 79 84 89 97 105)

# Log file where the output will be appended
logfile="output_xyzrl.log"

# Clear the logfile if it exists (optional)
> "$logfile"

# Loop through each value and execute the command
# nl loop
# for i in "${rvalues[@]}"; do
#   for j in "${lvalues[@]}"; do
#     echo "Running: ./simple -r $i -l $j"
#     ./simple -r "$i" -l "$j" >> "$logfile"
#   done
# done

for i in "${rvalues[@]}"; do
  for j in "${lvalues[@]}"; do
    for k in "${xyzvalues[@]}"; do
      echo "Running: ./simple -r $i -l $j -n $k"
      ./simple -r "$i" -l "$j" -n "$k" >> "$logfile"
    done
  done
done
