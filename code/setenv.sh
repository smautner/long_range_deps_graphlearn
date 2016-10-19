#!/bin/bash
CODEPATH="$PWD/scratch/nips2016/code"
#CODEPATH="$PWD"
#PYTHONPATH="/home/ikea/miniconda2/lib/python2.7/site-packages:$PYTHONPATH"
PYTHONPATH="$CODEPATH/deps/GraphLearn:$PYTHONPATH"
PYTHONPATH="$CODEPATH/deps/EDeN:$PYTHONPATH"
PYTHONPATH="$CODEPATH/deps:$PYTHONPATH"
#MPLCONFIGDIR="/home/mautner/mylittlepony/matplotlib$SGE_TASK_ID/crap"; export MPLCONFIGDIR

export  CODEPATH
export  PYTHONPATH
