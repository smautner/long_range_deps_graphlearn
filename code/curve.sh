#!/bin/bash
#$ -l h_vmem=6G
#echo "##################################"
#echo "#  CHECK PATHS IN THIS FILE       "
#echo "##################################"
CODEPATH="$PWD/scratch/nips2016/code"
PYTHONPATH="$CODEPATH/deps/GraphLearn:$PYTHONPATH"
PYTHONPATH="$CODEPATH/deps/EDeN:$PYTHONPATH"
MPLCONFIGDIR="/home/mautner/mylittlepony/matplotlib$SGE_TASK_ID/crap"

# we need to do this becuase the cluster sucks with env vars :(
export PYTHONPATH
export CODEPATH
echo "i hope the cpu option works :) "
cd "$CODEPATH/notebooks"; python -m cProfile -o MYPROFILE curve.py 1

