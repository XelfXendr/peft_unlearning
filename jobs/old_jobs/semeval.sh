#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:gpu_mem=64gb:scratch_local=64gb
#PBS -l walltime=8:00:00
#PBS -N unlearn_submission

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
cd src/semeval
$SCRATCHDIR/llm_thesis/venv/bin/python submission.py \
    --load_dir='/storage/brno2/home/bronecja/llm_thesis_logs/unlearn.py-2025-01-10_034215-bs=4,b=1.0,d=None,e=20,ee=50000,km=0.5,lr=0.0001,lme=-1,lr=5,m=7B,nm=1.0,rm=1.0,sln=True,sm=True,s=42,t=4/model' \
    --retain_dir='/storage/brno2/home/bronecja/llm_thesis/src/semeval25-unlearning-data/data' \
    --forget_dir='/storage/brno2/home/bronecja/llm_thesis/src/semeval25-unlearning-data/data' \
    --output_dir='/storage/brno2/home/bronecja/test_model'

