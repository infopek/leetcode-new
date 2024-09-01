#!/bin/bash

# https://leetcode.com/problems/valid-phone-numbers/description/
filename=$1
grep -E "^([0-9]{3}-|\([0-9]{3}\) )[0-9]{3}-[0-9]{4}$" "${filename}"
