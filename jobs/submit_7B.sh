#!/bin/bash

# try various values for beta and learning rate
for beta in 0.1 0.5 0.7 1.0 1.1
do
    # submit the job
    qsub -q gpu -v MODEL=7B,EP=20,BETA=$beta,LR=1e-4,NPO=1.,RT=1.,KL=0.5 unlearn.sh
done