#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:gpu_mem=32gb:scratch_local=64gb
#PBS -l walltime=24:00:00
#PBS -N semeval_task

# storage is shared via NFSv4
DATADIR="/storage/brno2/home/$LOGNAME/llm_thesis"
LOGDIR="/storage/brno2/home/$LOGNAME/llm_thesis_logs"

# sets the scratchdir as a temporary dir, bypassing the metacentre disk quota
export TMPDIR=$SCRATCHDIR
# clean the SCRATCH when job finishes
trap 'clean_scratch' TERM EXIT

# copy working directory into the scratchdir
cp -r $DATADIR $SCRATCHDIR
cd $SCRATCHDIR/llm_thesis

# set up python environment
module add python/python-3.10.4-intel-19.0.4-sc7snnf
python3 -m venv venv
venv/bin/pip install --no-cache-dir --upgrade pip setuptools
venv/bin/pip install --no-cache-dir -r requirements.txt

# ... the computation ...
cd src
$SCRATCHDIR/llm_thesis/venv/bin/python unlearn.py \
    --model=$MODEL \
    --logdir=$LOGDIR \
    --threads=4 \
    --batch_size=4 \
    --epochs=$EP \
    --evaluate_every=5 \
    --beta=$BETA \
    --learning_rate=$LR \
    --npo_mult=$NPO \
    --rt_mult=$RT \
    --kl_mult=$KL \
    --lora_rank=$RANK \
    --lora_merge_every=$MERGE
