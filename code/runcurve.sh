

mkdir -p /home/mautner/JOBZ_c4/curve_o
mkdir -p /home/mautner/JOBZ_c4/curve_e
source ./setenv.sh
#qsub -V -t 1-72 -pe smp 6  scratch/nips2016/code/curve.sh
qsub -V -t 1-110 curve.sh
