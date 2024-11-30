#!/bin/bash

# try various values for beta and learning rate
for beta in 0.1 0.5 0.1 0.15
do
    for lr in '1e-3' '1e-4' '1e-5'
    do
        # submit the job
        qsub -q gpu -v BETA=$beta,LR=$lr unlearn.sh
    done
done
