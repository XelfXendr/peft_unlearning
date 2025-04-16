#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:gpu_mem=16gb:scratch_local=64gb
#PBS -l walltime=24:00:00
#PBS -N 1B_job

# storage is shared via NFSv4
HOMEDIR="/storage/brno2/home/$LOGNAME"
DATADIR="$HOMEDIR/peft_unlearning"
LOGDIR="$HOMEDIR/1B_unlearning_logs"

MMLUDATA="$HOMEDIR/open-instruct/data/eval/mmlu"
SEMEVALDATA="$HOMEDIR/peft_unlearning/src/semeval25-unlearning-data"

UV="$HOMEDIR/.local/bin/uv"

HFTOKEN=$(cat "$DATADIR"/hf_token.txt)

# sets the scratchdir as a temporary dir, bypassing the metacentre disk quota
export TMPDIR=$SCRATCHDIR
# clean the SCRATCH when job finishes
trap 'clean_scratch' TERM EXIT

# copy working directory into the scratchdir
cp -r "$DATADIR" "$SCRATCHDIR"
cd "$SCRATCHDIR"/peft_unlearning || exit

# set up python environment
$UV init --bare
$UV add -r requirements.txt --cache-dir "$SCRATCHDIR/uv_cache"

# ... the computation ...
cd src || exit
$UV run unlearn.py \
    --hf_token="$HFTOKEN" \
    --model="$MODEL" \
    --logdir="$LOGDIR" \
    --threads=4 \
    --batch_size=4 \
    --epochs="$EP" \
    --evaluate_every=-1 \
    --learning_rate="$LR" \
    --beta="$BETA" \
    --npo_mult="$NPO" \
    --rt_mult="$RT" \
    --kl_mult="$KL" \
    --lora_rank="$RANK" \
    --lora_alpha="$ALPHA" \
    --lora_merge_every=-1 \
    --save_every="$SAVE" \
    --save_model \
    --save_logdir_name \
    --seed="$SEED"

RUNLOGDIR=$(<logdir.txt)

echo "Evaluating MMLU"
cd ../..
# download MMLU eval framework
git clone https://github.com/allenai/open-instruct.git
cd open-instruct || exit
git checkout 74897429b291acc2e579a888348c6f185cbdbec5
$UV sync --cache-dir "$SCRATCHDIR/uv_cache"
$UV sync --extra compile --cache-dir "$SCRATCHDIR/uv_cache" # to install flash attention

export PYTHONPATH=$PYTHONPATH:$(pwd)

for f in $(find "$RUNLOGDIR" -type f -name 'model-*' | grep -o "\(.*\)/" | sort -u)
do
    $UV run eval/mmlu/run_eval.py \
        --model_name_or_path="$f" \
        --tokenizer_name_or_path="$f" \
        --save_dir="$f" \
        --data_dir="$MMLUDATA"
done

echo "Evaluating SEMEVAL"
cd ../peft_unlearning || exit
for f in $(find "$RUNLOGDIR" -type f -name 'model-*' | grep -o "\(.*\)/" | sort -u)
do
    $UV run semeval/eval/semeval_evaluation.py \
        --checkpoint_path="$f" \
        --mmlu_metrics_file_path="$f"/metrics.json \
        --data_path="$SEMEVALDATA"/data/ \
        --mia_data_path="$SEMEVALDATA"/mia_data/ \
        --output_dir="$f"
done

