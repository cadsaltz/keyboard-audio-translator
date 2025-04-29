#!/bin/bash

for file in *.wav; do
	python3 train.py "$file"
	echo "Added file: " "$file"
done
