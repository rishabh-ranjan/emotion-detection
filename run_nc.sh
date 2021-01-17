#!/bin/bash

if [[ "$1" -eq "1" ]]
then
	python3 a.py "$2" "$3" "$4"
elif [[ "$2" -eq "2" ]]
then
	python3 b.py "$2" "$3" "$4"
else
	python3 c.py "$2" "$3" "$4"
fi
