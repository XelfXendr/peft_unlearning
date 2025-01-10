#!/bin/bash

for lr in 1e-4 2e-5
do
    for beta in 0.5 1.
    do
        for rank in 2 5 10
        do
            # submit the job
            qsub -q gpu -v MODEL=7B,EP=20,BETA=$beta,LR=$lr,NPO=1.,RT=1.,KL=0.5,MERGE=-1,RANK=$rank unlearn_eval.sh
        done
    done
done