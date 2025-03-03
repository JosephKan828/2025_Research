#!/bin/sh

for y in {2006..2017}; do
    python merge_CSdata.py "$y"
done