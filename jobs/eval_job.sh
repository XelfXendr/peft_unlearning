#!/bin/bash

qsub -q gpu -v MODEL=1B,EP=0,BETA=0.5,LR=1e-4,NPO=1.,RT=1.,KL=0.5,MERGE=-1,RANK=5 unlearn_eval.sh