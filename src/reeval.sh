#!/bin/bash

for f in $(find "/storage/brno2/home/bronecja/llm_thesis_logs" -maxdepth 1)
do
   if [ -d $f'/model' ]; then
      qsub -q gpu -v RUNLOGDIR="'$f'" just_eval.sh
   fi
done
