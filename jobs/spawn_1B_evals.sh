#!/bin/bash

for rundir in $(ls /storage/brno2/home/bronecja/1B_unlearning_logs/); do
    rundir="/storage/brno2/home/bronecja/1B_unlearning_logs/$rundir"
    qsub -q gpu -v RUNLOGDIR="'$rundir'" eval_1Bs.sh
done