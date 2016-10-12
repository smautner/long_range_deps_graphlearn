
CODEPATH="$PWD/scratch/nips2016/code"
PYTHONPATH="$CODEPATH/deps/GraphLearn:$PYTHONPATH"
PYTHONPATH="$CODEPATH/deps/EDeN:$PYTHONPATH"
MPLCONFIGDIR="/home/mautner/mylittlepony/matplotlib$SGE_TASK_ID/crap"

# we need to do this becuase the cluster sucks with env vars :(
export PYTHONPATH
export CODEPATH
export MPLCONFIGDIR

qsub -V -t 1-72 -l cpu=6 scratch/nips2016/code/curve.sh
