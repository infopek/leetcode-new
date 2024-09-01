#!/bin/bash

# https://leetcode.com/problems/tenth-line/

filename=$1

# Solution #1
count=0
while IFS= read -r line; do
    count=$((count + 1))
    if [[ count -eq 10 ]]; then
        echo "${line}"
        break
    fi
done < "${filename}"

# Solution #2
sed "10q;d" "${filename}"

# Solution #3
head -10 "${filename}" | tail -1  # prints the last line if there are less than 10 lines!

# Solution #4
awk "NR == 10 {print; exit}" "${filename}"

read -n 1