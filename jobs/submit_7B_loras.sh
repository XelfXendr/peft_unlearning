#!/bin/bash

# try various values for beta and learning rate
for merge in 100 1000 10000
do
    for rank in 1 2 5 10
    do
        # submit the job
        qsub -q gpu -v MODEL=7B,EP=30,BETA=0.5,LR=1e-4,NPO=1.,RT=1.,KL=0.5,MERGE=$merge,RANK=$rank unlearn-7B.sh
    done
done