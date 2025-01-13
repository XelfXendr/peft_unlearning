#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:gpu_mem=32gb:scratch_local=64gb
#PBS -l walltime=8:00:00
#PBS -N just_eval

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

echo "Starting"
echo $RUNLOGDIR

# set up python environment
module add python/python-3.10.4-intel-19.0.4-sc7snnf
python3 -m venv venv
venv/bin/pip install --no-cache-dir --upgrade pip setuptools

# ... the computation ...
cd src

# download MMLU eval framework
git clone https://github.com/allenai/open-instruct.git
cd open-instruct

# install requirements
$SCRATCHDIR/llm_thesis/venv/bin/python -m pip install --upgrade pip "setuptools<70.0.0" wheel 
$SCRATCHDIR/llm_thesis/venv/bin/python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
$SCRATCHDIR/llm_thesis/venv/bin/python -m pip install packaging
$SCRATCHDIR/llm_thesis/venv/bin/python -m pip install flash-attn==2.6.3 --no-build-isolation
$SCRATCHDIR/llm_thesis/venv/bin/python -m pip install -r requirements.txt
$SCRATCHDIR/llm_thesis/venv/bin/python -m nltk.downloader punkt
$SCRATCHDIR/llm_thesis/venv/bin/python -m pip install -U transformers

export PYTHONPATH=$PYTHONPATH:$(pwd)
scripts/data/prepare_eval_data.sh

$SCRATCHDIR/llm_thesis/venv/bin/python3 eval/mmlu/run_eval.py \
    --model_name_or_path=$RUNLOGDIR/model \
    --tokenizer_name_or_path=$RUNLOGDIR/model \
    --save_dir=$RUNLOGDIR \
    --data_dir=$SCRATCHDIR/llm_thesis/src/open-instruct/data/eval/mmlu

cd ..
$SCRATCHDIR/llm_thesis/venv/bin/python3 scripts/semeval_evaluation.py \
    --checkpoint_path=$RUNLOGDIR/model \
    --mmlu_metrics_file_path=$RUNLOGDIR/metrics.json \
    --data_path=semeval25-unlearning-data/data/ \
    --mia_data_path=semeval25-unlearning-data/mia_data/ \
    --output_dir=$RUNLOGDIR