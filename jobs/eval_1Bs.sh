#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:gpu_mem=16gb:scratch_local=64gb
#PBS -l walltime=8:00:00
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

echo "MMLU Eval"

git clone https://github.com/allenai/open-instruct.git
cd open-instruct
git checkout 74897429b291acc2e579a888348c6f185cbdbec5
$UV sync --cache-dir "$SCRATCHDIR/uv_cache"
$UV sync --extra compile --cache-dir "$SCRATCHDIR/uv_cache" # to install flash attention

export PYTHONPATH=$PYTHONPATH:$(pwd)

for f in $(find "$RUNLOGDIR" -type f -name 'model-*' | grep -o "\(.*\)/" | sort -u)
do
    if [ ! -f "$f"/metrics.json ]; then
        $UV run eval/mmlu/run_eval.py \
            --model_name_or_path="$f" \
            --tokenizer_name_or_path="$f" \
            --save_dir="$f" \
            --data_dir="$MMLUDATA"
    fi
done

echo "SemEval eval"

cd ..
git clone https://github.com/XelfXendr/peft_unlearning.git
cd peft_unlearning
$UV init --bare
$UV add -r requirements.txt --cache-dir "$SCRATCHDIR/uv_cache"

mkdir semeval/eval/
cp "$DATADIR/semeval/eval/semeval_evaluation.py" semeval/eval/

for f in $(find "$RUNLOGDIR" -type f -name 'model-*' | grep -o "\(.*\)/" | sort -u)
do
    if [ ! -f "$f"/evaluation_results.jsonl ]; then
        $UV run semeval/eval/semeval_evaluation.py \
            --checkpoint_path="$f" \
            --mmlu_metrics_file_path="$f"/metrics.json \
            --data_path="$SEMEVALDATA"/data/ \
            --mia_data_path="$SEMEVALDATA"/mia_data/ \
            --output_dir="$f"
    fi
done
