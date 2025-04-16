#!/bin/bash
for _i in 0 1 2 3 4
do
    SEED=$(date +%N)
    qsub -q gpu -v MODEL=1B,EP=20,LR=1e-4,BETA=0.5,NPO=1.,RT=1.,KL=0.5,RANK=1,ALPHA=5,SEED="$SEED",SAVE=2 1B_job.sh
    qsub -q gpu -v MODEL=1B,EP=20,LR=1e-4,BETA=0.5,NPO=1.,RT=1.,KL=0.5,RANK=2,ALPHA=5,SEED="$SEED",SAVE=2 1B_job.sh
    qsub -q gpu -v MODEL=1B,EP=20,LR=1e-4,BETA=0.5,NPO=1.,RT=1.,KL=0.5,RANK=5,ALPHA=5,SEED="$SEED",SAVE=2 1B_job.sh
    qsub -q gpu -v MODEL=1B,EP=20,LR=1e-4,BETA=0.5,NPO=1.,RT=1.,KL=0.5,RANK=10,ALPHA=5,SEED="$SEED",SAVE=2 1B_job.sh
    qsub -q gpu -v MODEL=1B,EP=20,LR=1e-4,BETA=0.5,NPO=1.,RT=1.,KL=0.5,RANK=25,ALPHA=5,SEED="$SEED",SAVE=2 1B_job.sh
    qsub -q gpu -v MODEL=1B,EP=20,LR=1e-4,BETA=0.5,NPO=1.,RT=1.,KL=0.5,RANK=100,ALPHA=5,SEED="$SEED",SAVE=2 1B_job.sh
done
