#!/bin/bash

> alphatimelog.txt
for ((a=1; a<=25; a++))
do
    output=$(./simple -a $a)
    last_line=$(echo "a = $a, $output" | tail -n 1)
    printf "a = $a\n"
    echo "$last_line"
    echo "a = $a, $last_line" >> alphatimelog.txt
done

