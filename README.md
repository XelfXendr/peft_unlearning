# LLM Unlearning using Parameter-efficient Finetuning

This repository contains the solution to the [SemEval-2025 Task 4 on Unlearning sensitive content from Large Language Models](https://llmunlearningsemeval2025.github.io)

The [semeval/submissions/](semeval/submissions/) directory contains the submitted solutions to the challenge. 

# Setup

Using [uv](https://docs.astral.sh/uv/) (recommended):
```bash
uv sync
```

The `requirements.txt` file is also present for pip setup.

# Running the project
All available arguments may be listed using:
```bash
uv run src/unlearn.py --help
```

Important arguments:
```bash
  --hf_token HF_TOKEN   SemEval-2025 Task 4 hugging face token. (Required)
  --model {1B,7B}       Model to use.
  --logdir LOGDIR       Logdir.

  --batch_size BATCH_SIZE
                        Batch size.
  --epochs EPOCHS       Number of unlearning epochs.
  --learning_rate LEARNING_RATE
                        Learning rate.

  --beta BETA           Beta for NPO loss.
  --npo_mult NPO_MULT   NPO forget loss multiplier.
  --rt_mult RT_MULT     NLL retain loss multiplier.
  --kl_mult KL_MULT     KL divergence retain loss multiplier.

  --lora_rank LORA_RANK
                        Rank of the LoRAs.
                        Merge LoRAs every n batches (Experimental). `-1` means never.
  
  --evaluate_every EVALUATE_EVERY
                        Evaluate every n epochs. `-1` means never.
  --save_every SAVE_EVERY
                        Save checkpoint every n epochs. `-1` means never.
  --save_model, --no-save_model
                        Save model after training.
```