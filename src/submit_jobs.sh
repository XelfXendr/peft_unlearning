#!/bin/bash

# try various values for beta and learning rate
for beta in 1.0 1.1
do
    for lr in 1e-4 1e-5
    do
        # submit the job
        qsub -q gpu -v BETA=$beta,LR=$lr unlearn.sh
    done
done
