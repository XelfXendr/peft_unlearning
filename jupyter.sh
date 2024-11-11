#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16gb:scratch_local=32gb -I
#PBS -l walltime=24:00:00
#PBS -N semeval_task

echo "Loading"

# storage is shared via NFSv4
DATADIR="/storage/brno2/home/$LOGNAME/llm_thesis"
LOGDIR="/storage/brno2/home/$LOGNAME/llm_thesis/logs"

# sets the scratchdir as a temporary dir, bypassing the metacentre disk quota
export TMPDIR=$SCRATCHDIR
# clean the SCRATCH when job finishes
trap 'clean_scratch' TERM EXIT

# copy working directory into the scratchdir
cp -r $DATADIR $SCRATCHDIR
cd $SCRATCHDIR

echo "CDed"

# set up python environment
module add python/python-3.10.4-intel-19.0.4-sc7snnf
python3 -m venv venv
venv/bin/pip install --no-cache-dir --upgrade pip setuptools
venv/bin/pip install --no-cache-dir -r llm_thesis/requirements.txt

echo "Installed"

venv/bin/python -m jupyter notebook --port 8308