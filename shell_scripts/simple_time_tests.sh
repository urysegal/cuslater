#!/bin/bash

> mylog.txt
for ((a=6; a<=25; a++))
do
    output=$(./simple -a $a)
    last_line=$(echo "a = $a, $output" | tail -n 1)
    printf "a = $a\n"
    echo "$last_line"
    echo "a = $a, $last_line" >> mylog.txt
done

