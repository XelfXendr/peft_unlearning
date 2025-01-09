# Jan Bronec's master's thesis repository

---

# Notes

- lora merging while training doesn't work because I'm using the backbone for NPO loss computation (merging changes loss)

```bash
git clone git@github.com:allenai/open-instruct.git

cd open-instruct

$SCRATCHDIR/llm_thesis/venv/bin/python -m pip install --upgrade pip "setuptools<70.0.0" wheel 
# TODO, unpin setuptools when this issue in flash attention is resolved
$SCRATCHDIR/llm_thesis/venv/bin/python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
$SCRATCHDIR/llm_thesis/venv/bin/python -m pip install packaging
$SCRATCHDIR/llm_thesis/venv/bin/python -m pip install flash-attn==2.6.3 --no-build-isolation
$SCRATCHDIR/llm_thesis/venv/bin/python -m pip install -r requirements.txt
$SCRATCHDIR/llm_thesis/venv/bin/python -m nltk.downloader punkt
$SCRATCHDIR/llm_thesis/venv/bin/python -m pip install -U transformers

export PYTHONPATH=$PYTHONPATH:$(pwd)

scripts/data/prepare_eval_data.sh


cd open-instruct/eval/mmlu/
$SCRATCHDIR/llm_thesis/venv/bin/python3 run_eval.py \
    --model_name_or_path=$SCRATCHDIR/llm_thesis/src/model \
    --tokenizer_name_or_path=$SCRATCHDIR/llm_thesis/src/model \
    --save_dir=. \
    --data_dir=$SCRATCHDIR/llm_thesis/open-instruct/data/eval/mmlu

../../venv/bin/python3 evaluate_generations.py --debug --checkpoint_path=../model --mmlu_metrics_file_path=../metrics.json --data_path=data/ --output_dir=./ --mia_data_path=mia_data/
```

forget_train-00000-of-00001.parquet
forget_validation-00000-of-00001.parquet
retain.jsonl
retain_train-00000-of-00001.parquet
retain_validation-00000-of-00001.parquet