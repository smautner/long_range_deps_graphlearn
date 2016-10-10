#!/bin/bash
#$ -l h_vmem=6G
echo "##################################"
echo "#  CHECK PATHS IN THIS FILE       "
echo "##################################"

codepath=$PWD
PYTHONPATH="$codepath/deps/GraphLearn:$PYTHONPATH"
PYTHONPATH="$codepath/deps/EDeN:$PYTHONPATH"
echo $PYTHONPATH
cd notebooks; python curve.py $1

