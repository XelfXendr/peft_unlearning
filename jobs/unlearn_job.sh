#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:gpu_mem=64gb:scratch_local=64gb
#PBS -l walltime=24:00:00
#PBS -N unlearn_job

# storage is shared via NFSv4
HOMEDIR="/storage/brno2/home/$LOGNAME"
DATADIR="$HOMEDIR/peft_unlearning"
LOGDIR="$HOMEDIR/peft_unlearning_logs"

UV="$HOMEDIR/.local/bin/uv"

HFTOKEN=$(cat $DATADIR/hf_token.txt)

# sets the scratchdir as a temporary dir, bypassing the metacentre disk quota
export TMPDIR=$SCRATCHDIR
# clean the SCRATCH when job finishes
trap 'clean_scratch' TERM EXIT

# copy working directory into the scratchdir
cp -r $DATADIR $SCRATCHDIR
cd $SCRATCHDIR/peft_unlearning

# set up python environment
$UV init --bare
$UV add -r requirements.txt

# ... the computation ...
cd src
$UV run unlearn.py \
    --hf_token=$HFTOKEN \
    --model=$MODEL \
    --logdir=$LOGDIR \
    --threads=4 \
    --batch_size=4 \
    --epochs=$EP \
    --evaluate_every=-1 \
    --learning_rate=$LR \
    --beta=$BETA \
    --npo_mult=$NPO \
    --rt_mult=$RT \
    --kl_mult=$KL \
    --lora_rank=$RANK \
    --lora_merge_every=-1 \
    --save_every=$SAVE \
    --save_model=True \
    --save_logdir_name=True \
    --seed=$SEED

RUNLOGDIR=$(<logdir.txt)

# evaluate checkpoints
for f in $(find $RUNLOGDIR -type f -name 'model-*' | grep -o "\(.*\)/" | sort -u)
do
    qsub -q gpu -v MODELPATH="'$f'" $DATADIR/jobs/eval_model.sh
done