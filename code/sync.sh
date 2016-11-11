#!/bin/bash
git commit -a -m 'autosync'
git push
ssh cluster 'cd /scratch/bi01/mautner/nips2016; git pull'
ssh bipc    'cd /scratch/1/mautner/nips2016; git pull'

