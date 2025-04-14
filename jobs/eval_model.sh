#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:gpu_mem=32gb:scratch_local=64gb
#PBS -l walltime=4:00:00
#PBS -N eval_model

# storage is shared via NFSv4
HOMEDIR="/storage/brno2/home/$LOGNAME"
DATADIR="$HOMEDIR/peft_unlearning"
LOGDIR="$HOMEDIR/peft_unlearning_logs"

MMLUDATA="$HOMEDIR/open-instruct/data/eval/mmlu"
SEMEVALDATA="$HOMEDIR/peft_unlearning/src/semeval25-unlearning-data"

UV="$HOMEDIR/.local/bin/uv"

# sets the scratchdir as a temporary dir, bypassing the metacentre disk quota
export TMPDIR=$SCRATCHDIR
# clean the SCRATCH when job finishes
trap 'clean_scratch' TERM EXIT

cd $SCRATCHDIR

# download MMLU eval framework
if [ ! -f "$MODELPATH"/metrics.json ]; then
    # download MMLU eval framework
    git clone https://github.com/allenai/open-instruct.git
    cd open-instruct
    git checkout 74897429b291acc2e579a888348c6f185cbdbec5
    $UV sync
    $UV sync --extra compile # to install flash attention

    export PYTHONPATH=$PYTHONPATH:$(pwd)
    $UV run eval/mmlu/run_eval.py \
        --model_name_or_path=$MODELPATH \
        --tokenizer_name_or_path=$MODELPATH \
        --save_dir=$MODELPATH \
        --data_dir=$MMLUDATA

    cd ..
fi

if [ ! -f "$MODELPATH"/evaluation_results.jsonl ]; then
    git clone https://github.com/XelfXendr/peft_unlearning.git
    cd peft_unlearning
    $UV init --bare
    $UV add -r requirements.txt

    $UV run semeval/eval/semeval_evaluation.py \
        --checkpoint_path=$MODELPATH \
        --mmlu_metrics_file_path=$MODELPATH/metrics.json \
        --data_path=$SEMEVALDATA/data/ \
        --mia_data_path=$SEMEVALDATA/mia_data/ \
        --output_dir=$MODELPATH
fi
