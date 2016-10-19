#!/bin/bash
source ./setenv.sh

if [ ! $# -eq 1 ] 
then
       echo " USAGE ./start_nb.sh portnumber" 
       exit 
fi
jupyter notebook  --no-browser --port $1

