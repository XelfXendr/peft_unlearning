#!/bin/bash

# try various values for beta and learning rate
qsub -q gpu -v MODEL=7B,EP=10,BETA=0.5,LR=1e-4,NPO=1.,RT=1.,KL=0.5,MERGE=-1,RANK=5 unlearn-7B.sh