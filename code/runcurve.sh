
CODEPATH="$PWD/scratch/nips2016/code"
PYTHONPATH="$codepath/deps/GraphLearn:$PYTHONPATH"
PYTHONPATH="$codepath/deps/EDeN:$PYTHONPATH"
MPLCONFIGDIR="/home/mautner/mylittlepony/matplotlib$SGE_TASK_ID/crap"



# we need to do this becuase the cluster sucks with env vars :(
export PYTHONPATH
export CODEPATH
export MPLCONFIGDIR

qsub -V -t 1-72 scratch/nips2016/code/curve.sh
