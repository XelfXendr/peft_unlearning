#!/bin/bash
for i in 0 1 2 3 4
do
    SEED=$(date +%s%N)
    qsub -q gpu -v MODEL=7B,EP=20,LR=1e-4,BETA=0.5,NPO=1.,RT=1.,KL=0.5,RANK=5,SEED=$SEED unlearn_eval.sh
    qsub -q gpu -v MODEL=7B,EP=20,LR=1e-4,BETA=0.5,NPO=1.,RT=1.,KL=0,RANK=5,SEED=$SEED unlearn_eval.sh
    qsub -q gpu -v MODEL=7B,EP=20,LR=1e-4,BETA=0.5,NPO=1.,RT=1.,KL=1.,RANK=5,SEED=$SEED unlearn_eval.sh
    qsub -q gpu -v MODEL=7B,EP=20,LR=1e-4,BETA=0.5,NPO=1.,RT=0.,KL=1.,RANK=5,SEED=$SEED unlearn_eval.sh
done