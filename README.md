# SLLM

This repository provides code for training and evaluating Tree-Planted Transformers (TPTs).

> [Tree-Planted Transformers: Unidirectional Transformer Language Models with Implicit Syntactic Supervision](https://arxiv.org/abs/2402.12691)<br>
> Ryo Yoshida and Taiga Someya and Yohei Oseki<br>
> ACL 2024 (Findings) <br>
>

## Requirement
- python = ">=3.9,<3.13"

## Installation
```bash
git clone https://github.com/osekilab/TPT.git
cd TPT
poetry install
poetry shell
```

## Data preparation
```bash
python src/preprocess_dependency.py \
    --train_file_path ./data/train.txt \
    --val_file_path ./data/valid.txt \
    --test_file_path ./data/test.txt \
    --output_dir ./data/preprocessed/dependency/ \
    --convert_method exponential \
    --min_n_terminals 3 \
    --seed 42
```

## Training
- Setup
    ```bash
    accelerate config
    ```
- Train
    ```bash
    accelerate launch src/train.py \
        --train_file ./data/preprocessed/dependency/train.sequential=False.random=False.convert_method=exponential.jsonl \
        --validation_file ./data/preprocessed/dependency/val.sequential=False.random=False.convert_method=exponential.jsonl \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 10 \
        --output_dir ./experiments/example/ \
        --seed 42 \
        --checkpointing_steps 5000 \
        --with_tracking \
        --report_to tensorboard \
        --attn_loss_weight 0.5 \
        --attn_loss_layers [11] \
        --attn_loss_heads [0] \
        --attn_loss_reduction "none"
    ```

## Evaluation
```bash
python3 src/calc_surp.py \
    --test_file ./data/test_tokens.txt \
    --model_path ./experiments/example/ \
    --output_path ./experiments/example/perplexity.txt \
    --gpu_id 0 \
    --treebank_tokenized
```

## Credits
`src/train.py` is based on `run_clm_trainer.py` from [Huggingface Transformers](https://github.com/huggingface/transformers).

## Note
If you want to download TPTs trained in our paper, please contact `yoshiryo0617 [at] g.ecc.u-tokyo.ac.jp`.
