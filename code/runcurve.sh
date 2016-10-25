

source ./setenv.sh
#qsub -V -t 1-72 -pe smp 6  scratch/nips2016/code/curve.sh
qsub -V -t 1-10 curve.sh $TASK_ID
