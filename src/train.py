#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications copyright 2024 Ryo Yoshida
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments. # noqa

import argparse
import json
import logging
import math
import os
import random

# from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset

# from huggingface_hub import Repository, create_repo
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (  # CONFIG_MAPPING,; MODEL_MAPPING,; AutoConfig,; AutoModelForCausalLM,; AutoTokenizer,; default_data_collator, # noqa
    GPT2Config,
    GPT2LMHeadModel,
    SchedulerType,
    get_scheduler,
)

from preprocess_constituency import (  # we use the constieuency version which contains more tokens # noqa
    get_gpt2tokenizer,
)

# from itertools import chain


# from transformers.utils import check_min_version, send_example_telemetry
# from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks. # noqa
# check_min_version("4.35.0.dev0")

logger = get_logger(__name__)

# require_version(
#     "datasets>=1.8.0",
#     "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
# )

# MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Define the type for the attention tuples for clarity
AttentionTupleType = Tuple[torch.Tensor, ...]


def _pad_to_feature_dim(
    tensor: torch.Tensor,
    target_feature_dim: int,
    padding_value: Union[int, bool] = -100,
) -> torch.Tensor:
    padding = torch.full(
        (tensor.shape[0], target_feature_dim - tensor.shape[1]), padding_value
    )
    return torch.cat([tensor, padding], dim=1)


def _pad_tensors(
    tensors: List[torch.Tensor], padding_value: Union[int, bool] = -100
) -> torch.Tensor:
    max_feature_dim = max(tensor.shape[1] for tensor in tensors)

    tensors_padded = [
        _pad_to_feature_dim(tensor, max_feature_dim, padding_value)
        for tensor in tensors
    ]
    return pad_sequence(tensors_padded, batch_first=True, padding_value=padding_value)


def collate_fn_builder(pad_id: int = 50256):
    def collate_fn(batch: Dict[str, torch.Tensor]):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        attn_matrix = [torch.tensor(item["attn_matrix"]) for item in batch]
        row_word_token_membership_mask = [
            torch.tensor(item["row_word_token_membership_mask"]) for item in batch
        ]
        col_word_token_membership_mask = [
            torch.tensor(item["col_word_token_membership_mask"]) for item in batch
        ]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        attn_matrix_padded = _pad_tensors(
            attn_matrix, padding_value=0
        )  # pad_id for attn_matrix is 0; which is ignored in loss calculation
        row_word_token_membership_mask_padded = _pad_tensors(
            row_word_token_membership_mask, padding_value=False
        )  # pad_id for row_word_token_membership_mask is False; which is ignored in attentions compression # noqa
        col_word_token_membership_mask_padded = _pad_tensors(
            col_word_token_membership_mask, padding_value=False
        )  # pad_id for col_word_token_membership_mask is False; which is ignored in attentions compression # noqa
        labels = input_ids_padded.clone()
        labels[labels == pad_id] = -100

        return {
            "input_ids": input_ids_padded,
            "labels": labels,
            "attn_matrix": attn_matrix_padded,
            "row_word_token_membership_mask": row_word_token_membership_mask_padded.float(),  # noqa
            "col_word_token_membership_mask": col_word_token_membership_mask_padded.float(),  # noqa
        }

    return collate_fn


def _extract_layers_and_heads(
    attentions: AttentionTupleType,
    attn_loss_layers: List[int],
    attn_loss_heads: List[int],
) -> torch.Tensor:
    # Extract the attention of the specified layers and heads
    model_attn = torch.stack(
        [attentions[i] for i in attn_loss_layers]
    )  # (num_layers, batch_size, num_heads, seq_len, seq_len)
    model_attn = model_attn[:, :, attn_loss_heads, :, :]
    # Remove the attention for preceding words with last token
    model_attn = model_attn[
        ..., :-1, :-1
    ]  # (num_layers, batch_size, num_heads, seq_len-1, seq_len-1)
    return model_attn


def extract_lower_triangle_and_normalize_rows2one(tensor: torch.Tensor) -> torch.Tensor:
    # Extract the lower triangle of the last two dimensions
    lower_triangles = torch.tril(tensor, diagonal=0)

    # Flatten the tensor except for the last two dimensions and normalize
    shape = lower_triangles.size()
    lower_triangles_flat = lower_triangles.view(-1, shape[-1])
    row_sums = lower_triangles_flat.sum(axis=1, keepdim=True)
    row_sums[row_sums == 0] = 1
    normalized_flat = lower_triangles_flat / row_sums

    # Reshape back to the original shape
    return normalized_flat.view(shape)


def _calculate_loss(
    model_attn: torch.Tensor, attn_label: torch.Tensor, loss_fct: torch.nn.KLDivLoss
) -> torch.Tensor:
    # Avoid zero division and log domain error
    epsilon = 1e-30
    model_attn.clamp_(min=epsilon)
    attn_label.clamp_(min=epsilon)

    # Calculate point-wise loss
    attn_loss_pointwise = loss_fct(torch.log(model_attn), attn_label)

    # Create the padding mask and apply it
    padding_mask = attn_label != epsilon
    attn_loss_pointwise *= padding_mask.float()

    # Normalize the loss
    padding_mask_flat = padding_mask.view(-1, padding_mask.shape[-1])  # (-1, seq_len-1)
    n_attn_distribution = torch.sum(
        (padding_mask_flat != 0).any(dim=-1)
    ).item()  # contains at least one non-padding token
    attn_loss = attn_loss_pointwise.sum() / n_attn_distribution

    return attn_loss


def compute_attn_loss_builder(
    attn_loss_reduction: str,
) -> Callable[[AttentionTupleType, torch.Tensor, List[int], List[int]], torch.Tensor]:
    loss_fct = torch.nn.KLDivLoss(reduction="none")

    def compute_layer_head_mean_attn_loss(
        attentions: AttentionTupleType,
        attn_matrix: torch.Tensor,
        row_word_token_membership_mask: torch.Tensor,
        col_word_token_membership_mask: torch.Tensor,
        attn_loss_layers: List[int],
        attn_loss_heads: List[int],
    ) -> torch.Tensor:
        model_attn = _extract_layers_and_heads(
            attentions, attn_loss_layers, attn_loss_heads
        )  # (num_layers, batch_size, num_heads, token_len-1, token_len-1)
        # Mean over heads and layers
        model_attn = torch.mean(
            model_attn, dim=[0, 2]
        )  # (batch_size, token_len-1, token_len-1)

        # compress the model_attn to match the attn_matrix
        model_attn_compressed = torch.einsum(
            "bij,bki,blj->bkl",
            model_attn,
            row_word_token_membership_mask,
            col_word_token_membership_mask,
        )  # (batch_size, word_len-1, word_len-1)
        model_attn_compressed_extracted_normalized = (
            extract_lower_triangle_and_normalize_rows2one(model_attn_compressed)
        )

        attn_label = attn_matrix
        assert model_attn_compressed_extracted_normalized.shape == attn_label.shape
        return _calculate_loss(
            model_attn_compressed_extracted_normalized, attn_label, loss_fct
        )

    def compute_layer_mean_attn_loss(
        attentions: AttentionTupleType,
        attn_matrix: torch.Tensor,
        row_word_token_membership_mask: torch.Tensor,
        col_word_token_membership_mask: torch.Tensor,
        attn_loss_layers: List[int],
        attn_loss_heads: List[int],
    ) -> torch.Tensor:
        model_attn = _extract_layers_and_heads(
            attentions, attn_loss_layers, attn_loss_heads
        )  # (num_layers, batch_size, num_heads, token_len-1, token_len-1)
        # Mean over layers
        model_attn = torch.mean(
            model_attn, dim=0
        )  # (batch_size, num_heads, token_len-1, token_len-1)

        # compress the model_attn to match the attn_matrix
        row_word_token_membership_mask = row_word_token_membership_mask.unsqueeze(
            1
        ).repeat(
            1, model_attn.shape[1], 1, 1
        )  # (batch_size, num_heads, word_len-1, token_len-1) # noqa
        col_word_token_membership_mask = col_word_token_membership_mask.unsqueeze(
            1
        ).repeat(
            1, model_attn.shape[1], 1, 1
        )  # (batch_size, num_heads, word_len-1, token_len-1) # noqa
        model_attn_compressed = torch.einsum(
            "bhij,bhki,bhlj->bhkl",
            model_attn,
            row_word_token_membership_mask,
            col_word_token_membership_mask,
        )  # (batch_size, word_len-1, word_len-1)
        model_attn_compressed_extracted_normalized = (
            extract_lower_triangle_and_normalize_rows2one(model_attn_compressed)
        )

        attn_label = attn_matrix.unsqueeze(1).repeat(
            1, model_attn.shape[1], 1, 1
        )  # (batch_size, num_heads, word_len-1, word_len-1)
        assert model_attn_compressed_extracted_normalized.shape == attn_label.shape
        return _calculate_loss(
            model_attn_compressed_extracted_normalized, attn_label, loss_fct
        )

    def compute_head_mean_attn_loss(
        attentions: AttentionTupleType,
        attn_matrix: torch.Tensor,
        row_word_token_membership_mask: torch.Tensor,
        col_word_token_membership_mask: torch.Tensor,
        attn_loss_layers: List[int],
        attn_loss_heads: List[int],
    ) -> torch.Tensor:
        model_attn = _extract_layers_and_heads(
            attentions, attn_loss_layers, attn_loss_heads
        )  # (num_layers, batch_size, num_heads, token_len-1, token_len-1)
        model_attn = torch.mean(
            model_attn, dim=2
        )  # (num_layers, batch_size, token_len-1, token_len-1)

        # compress the model_attn to match the attn_matrix
        row_word_token_membership_mask = row_word_token_membership_mask.unsqueeze(
            0
        ).repeat(
            model_attn.shape[0], 1, 1, 1
        )  # (num_layers, batch_size, word_len-1, token_len-1) # noqa
        col_word_token_membership_mask = col_word_token_membership_mask.unsqueeze(
            0
        ).repeat(
            model_attn.shape[0], 1, 1, 1
        )  # (num_layers, batch_size, word_len-1, token_len-1) # noqa
        model_attn_compressed = torch.einsum(
            "ybij,ybki,yblj->ybkl",
            model_attn,
            row_word_token_membership_mask,
            col_word_token_membership_mask,
        )  # (num_layers, batch_size, word_len-1, word_len-1)
        model_attn_compressed_extracted_normalized = (
            extract_lower_triangle_and_normalize_rows2one(model_attn_compressed)
        )

        attn_label = attn_matrix.unsqueeze(0).repeat(
            model_attn.shape[0], 1, 1, 1
        )  # (num_layers, batch_size, seq_len-1, seq_len-1)
        assert model_attn_compressed_extracted_normalized.shape == attn_label.shape
        return _calculate_loss(
            model_attn_compressed_extracted_normalized, attn_label, loss_fct
        )

    def compute_none_attn_loss(
        attentions: AttentionTupleType,
        attn_matrix: torch.Tensor,
        row_word_token_membership_mask: torch.Tensor,
        col_word_token_membership_mask: torch.Tensor,
        attn_loss_layers: List[int],
        attn_loss_heads: List[int],
    ) -> torch.Tensor:
        model_attn = _extract_layers_and_heads(
            attentions, attn_loss_layers, attn_loss_heads
        )  # (num_layers, batch_size, num_heads, token_len-1, token_len-1)

        # compress the model_attn to match the attn_matrix
        row_word_token_membership_mask = (
            row_word_token_membership_mask.unsqueeze(1)
            .repeat(
                1, model_attn.shape[2], 1, 1
            )  # (batch_size, num_heads, word_len-1, token_len-1)
            .unsqueeze(0)
            .repeat(
                model_attn.shape[0], 1, 1, 1, 1
            )  # (num_layers, batch_size, num_heads, word_len-1, token_len-1)
        )
        col_word_token_membership_mask = (
            col_word_token_membership_mask.unsqueeze(1)
            .repeat(
                1, model_attn.shape[2], 1, 1
            )  # (batch_size, num_heads, word_len-1, token_len-1)
            .unsqueeze(0)
            .repeat(
                model_attn.shape[0], 1, 1, 1, 1
            )  # (num_layers, batch_size, num_heads, word_len-1, token_len-1)
        )
        model_attn_compressed = torch.einsum(
            "ybhij,ybhki,ybhlj->ybhkl",
            model_attn,
            row_word_token_membership_mask,
            col_word_token_membership_mask,
        )
        model_attn_compressed_extracted_normalized = (
            extract_lower_triangle_and_normalize_rows2one(model_attn_compressed)
        )

        attn_label = (
            attn_matrix.unsqueeze(1)
            .repeat(
                1, model_attn.shape[2], 1, 1
            )  # (batch_size, num_heads, seq_len-1, seq_len-1)
            .unsqueeze(0)
            .repeat(model_attn.shape[0], 1, 1, 1, 1)
        )  # (num_layers, batch_size, num_heads, seq_len-1, seq_len-1)
        assert model_attn_compressed_extracted_normalized.shape == attn_label.shape
        return _calculate_loss(
            model_attn_compressed_extracted_normalized, attn_label, loss_fct
        )

    if attn_loss_reduction == "layer_head_mean":
        return compute_layer_head_mean_attn_loss
    elif attn_loss_reduction == "layer_mean":
        return compute_layer_mean_attn_loss
    elif attn_loss_reduction == "head_mean":
        return compute_head_mean_attn_loss
    elif attn_loss_reduction == "none":
        return compute_none_attn_loss
    else:
        raise ValueError(f"Unsupported attn_loss_reduction: {attn_loss_reduction}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    # parser.add_argument(
    #     "--dataset_name",
    #     type=str,
    #     default=None,
    #     help="The name of the dataset to use (via the datasets library).",
    # )
    # parser.add_argument(
    #     "--dataset_config_name",
    #     type=str,
    #     default=None,
    #     help="The configuration name of the dataset to use (via the datasets library).", # noqa
    # )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv, txt or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv, txt or a json file containing the validation data.",
    )
    # parser.add_argument(
    #     "--validation_split_percentage",
    #     default=5,
    #     help="The percentage of the train set used as validation set in case there's no validation split", # noqa
    # )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    # parser.add_argument(
    #     "--config_name",
    #     type=str,
    #     default=None,
    #     help="Pretrained config name or path if not the same as model_name",
    # )
    # parser.add_argument(
    #     "--tokenizer_name",
    #     type=str,
    #     default=None,
    #     help="Pretrained tokenizer name or path if not the same as model_name",
    # )
    # parser.add_argument(
    #     "--use_slow_tokenizer",
    #     action="store_true",
    #     help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).", # noqa
    # )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",  # noqa
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",  # noqa
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    # parser.add_argument(
    #     "--model_type",
    #     type=str,
    #     default=None,
    #     help="Model type to use if training from scratch.",
    #     choices=MODEL_TYPES,
    # )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"  # noqa
            " this size for training. Default to the model max input length for single sentence inputs (take into"  # noqa
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    # parser.add_argument(
    #     "--no_keep_linebreaks",
    #     action="store_true",
    #     help="Do not keep line breaks when using TXT files.",
    # )
    # parser.add_argument(
    #     "--push_to_hub",
    #     action="store_true",
    #     help="Whether or not to push the model to the Hub.",
    # )
    # parser.add_argument(
    #     "--hub_model_id",
    #     type=str,
    #     help="The name of the repository to keep in sync with the local `output_dir`.", # noqa
    # )
    # parser.add_argument(
    #     "--hub_token", type=str, help="The token to use to push to the Model Hub."
    # )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"  # noqa
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will"  # noqa
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",  # noqa
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'  # noqa
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'  # noqa
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."  # noqa
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument("--n_positions", type=int, default=1024)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_inner", type=int, default=None)
    parser.add_argument("--activation_function", type=str, default="gelu_new")
    parser.add_argument("--resid_pdrop", type=float, default=0.1)
    parser.add_argument("--embd_pdrop", type=float, default=0.1)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--layer_norm_epsilon", type=float, default=1e-5)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--attn_loss_weight", type=float, default=0.1)
    parser.add_argument("--attn_loss_layers", nargs="+", type=int, default=[9, 10, 11])
    parser.add_argument(
        "--attn_loss_heads",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    )
    parser.add_argument(
        "--attn_loss_reduction",
        type=str,
        default="layer_head_mean",
        choices=["layer_head_mean", "layer_mean", "head_mean", "none"],
    )
    args = parser.parse_args()

    # Sanity checks
    # if (
    #     args.dataset_name is None
    #     and args.train_file is None
    #     and args.validation_file is None
    # ):
    #     raise ValueError("Need either a dataset name or a training/validation file.")
    # else:
    #     if args.train_file is not None:
    #         extension = args.train_file.split(".")[-1]
    #         if extension not in ["csv", "json", "txt"]:
    #             raise ValueError("`train_file` should be a csv, json or txt file.")
    #     if args.validation_file is not None:
    #         extension = args.validation_file.split(".")[-1]
    #         if extension not in ["csv", "json", "txt"]:
    #             raise ValueError("`validation_file` should be a csv, json or txt file.") # noqa

    # if args.push_to_hub:
    #     if args.output_dir is None:
    #         raise ValueError(
    #             "Need an `output_dir` to create a repo when `--push_to_hub` is passed." # noqa
    #         )

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The # noqa
    # information sent is the one passed as arguments along with your Python/PyTorch versions. # noqa
    # send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example. # noqa
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers # noqa
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        # if args.push_to_hub:
        #     # Retrieve of infer repo_name
        #     repo_name = args.hub_model_id
        #     if repo_name is None:
        #         repo_name = Path(args.output_dir).absolute().name
        #     # Create repo and retrieve repo_id
        #     repo_id = create_repo(
        #         repo_name, exist_ok=True, token=args.hub_token
        #     ).repo_id
        #     # Clone repo locally
        #     repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token) # noqa

        #     with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
        #         if "step_*" not in gitignore:
        #             gitignore.write("step_*\n")
        #         if "epoch_*" not in gitignore:
        #             gitignore.write("epoch_*\n")
        # elif args.output_dir is not None:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info("Arguments:")
    logger.info(args)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below) # noqa
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/ # noqa
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called # noqa
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently # noqa
    # download the dataset.
    # if args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    #     if "validation" not in raw_datasets.keys():
    #         raw_datasets["validation"] = load_dataset(
    #             args.dataset_name,
    #             args.dataset_config_name,
    #             split=f"train[:{args.validation_split_percentage}%]",
    #         )
    #         raw_datasets["train"] = load_dataset(
    #             args.dataset_name,
    #             args.dataset_config_name,
    #             split=f"train[{args.validation_split_percentage}%:]",
    #         )
    # else:
    #     data_files = {}
    #     dataset_args = {}
    #     if args.train_file is not None:
    #         data_files["train"] = args.train_file
    #     if args.validation_file is not None:
    #         data_files["validation"] = args.validation_file
    #     extension = args.train_file.split(".")[-1]
    #     if extension == "txt":
    #         extension = "text"
    #         dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
    #     raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
    #     # If no validation data is there, validation_split_percentage will be used to divide the dataset. # noqa
    #     if "validation" not in raw_datasets.keys():
    #         raw_datasets["validation"] = load_dataset(
    #             extension,
    #             data_files=data_files,
    #             split=f"train[:{args.validation_split_percentage}%]",
    #             **dataset_args,
    #         )
    #         raw_datasets["train"] = load_dataset(
    #             extension,
    #             data_files=data_files,
    #             split=f"train[{args.validation_split_percentage}%:]",
    #             **dataset_args,
    #         )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at # noqa
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    data_files = {}
    data_files["train"] = args.train_file
    data_files["validation"] = args.validation_file
    raw_datasets = load_dataset("json", data_files=data_files)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently # noqa
    # download model & vocab.
    # if args.config_name:
    #     config = AutoConfig.from_pretrained(
    #         args.config_name,
    #         trust_remote_code=args.trust_remote_code,
    #     )
    # elif args.model_name_or_path:
    #     config = AutoConfig.from_pretrained(
    #         args.model_name_or_path,
    #         trust_remote_code=args.trust_remote_code,
    #     )
    # else:
    #     config = CONFIG_MAPPING[args.model_type]()
    #     logger.warning("You are instantiating a new config instance from scratch.")

    # if args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         args.tokenizer_name,
    #         use_fast=not args.use_slow_tokenizer,
    #         trust_remote_code=args.trust_remote_code,
    #     )
    # elif args.model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         args.model_name_or_path,
    #         use_fast=not args.use_slow_tokenizer,
    #         trust_remote_code=args.trust_remote_code,
    #     )
    # else:
    #     raise ValueError(
    #         "You are instantiating a new tokenizer from scratch. This is not supported by this script." # noqa
    #         "You can do it from another script, save it, and load it from here, using --tokenizer_name." # noqa
    #     )
    tokenizer = get_gpt2tokenizer()
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.n_positions,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_inner,
        activation_function=args.activation_function,
        resid_pdrop=args.resid_pdrop,
        embd_pdrop=args.embd_pdrop,
        attn_pdrop=args.attn_pdrop,
        layer_norm_epsilon=args.layer_norm_epsilon,
        initializer_range=args.initializer_range,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if args.model_name_or_path:
        # model = AutoModelForCausalLM.from_pretrained(
        #     args.model_name_or_path,
        #     from_tf=bool(".ckpt" in args.model_name_or_path),
        #     config=config,
        #     low_cpu_mem_usage=args.low_cpu_mem_usage,
        #     trust_remote_code=args.trust_remote_code,
        # )
        model = GPT2LMHeadModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        # model = AutoModelForCausalLM.from_config(
        #     config, trust_remote_code=args.trust_remote_code
        # )
        model = GPT2LMHeadModel(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch # noqa
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    # text_column_name = "text" if "text" in column_names else column_names[0]

    # def tokenize_function(examples):
    #     return tokenizer(examples[text_column_name])
    def remove_columns_function(examples):
        return {
            "token_ids": examples["token_ids"],
            "attn_matrix": examples["attn_matrix"],
            "word_token_membership_mask": examples["word_token_membership_mask"],
        }

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            # tokenize_function,
            remove_columns_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            # desc="Running tokenizer on dataset",
            desc="Removing columns",
        )

    if args.block_size is None:
        # block_size = tokenizer.model_max_length
        # if block_size > config.max_position_embeddings:
        #     logger.warning(
        #         f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). " # noqa
        #         f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx." # noqa
        #     )
        #     block_size = min(1024, config.max_position_embeddings)
        block_size = min(1024, config.max_position_embeddings)
    # else:
    #     if args.block_size > tokenizer.model_max_length:
    #         logger.warning(
    #             f"The block_size passed ({args.block_size}) is larger than the maximum length for the model" # noqa
    #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}." # noqa
    #         )
    #     block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size. # noqa
    # def group_texts(examples):
    #     # Concatenate all texts.
    #     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()} # noqa
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict. # noqa
    #     # We could add padding if the model supported it instead of this drop, you can customize this part to your needs. # noqa
    #     total_length = (total_length // block_size) * block_size
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result
    def _truncate_word_token_membership_mask(
        matrix: List[List[bool]], block_size: int
    ) -> List[List[int]]:
        np_matrix = np.array(matrix)
        col_truncated_matrix = np_matrix[:, :block_size]
        truncated_matrix = col_truncated_matrix[np.any(col_truncated_matrix, axis=1)]
        return truncated_matrix.tolist()

    def truncate_function(examples):
        truncated_input_ids = [seq[:block_size] for seq in examples["token_ids"]]
        truncated_word_token_membership_mask = [
            _truncate_word_token_membership_mask(matrix, block_size)
            for matrix in examples["word_token_membership_mask"]
        ]
        truncated_attn_matrix = [
            [
                # len(mask) -1 = word_size - 1
                # we remove the last word not depending on they are eos or not
                # this means that the last word will be ignored if all inside subwords are not truncated # noqa
                row[: len(mask) - 1]
                for row in attn_matrix[: len(mask) - 1]
            ]
            for attn_matrix, mask in zip(
                examples["attn_matrix"], truncated_word_token_membership_mask
            )
        ]

        truncated_row_word_token_membership_mask = [
            # truncate the first word and token because they will not be predicted
            # len(mask) = word_size
            # len(row) = token_size
            [row[1:] for row in mask[1:]]
            for mask in truncated_word_token_membership_mask
        ]
        truncated_col_word_token_membership_mask = [
            # truncate the last word and token because they will not be attended
            # len(mask) = word_size
            # len(row) = token_size
            [row[:-1] for row in mask[:-1]]
            for mask in truncated_word_token_membership_mask
        ]

        return {
            "input_ids": truncated_input_ids,
            "attn_matrix": truncated_attn_matrix,
            "row_word_token_membership_mask": truncated_row_word_token_membership_mask,  # noqa
            "col_word_token_membership_mask": truncated_col_word_token_membership_mask,  # noqa
        }

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder # noqa
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower # noqa
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information: # noqa
    # https://huggingface.co/docs/datasets/process#map

    with accelerator.main_process_first():
        #     lm_datasets = tokenized_datasets.map(
        #         group_texts,
        #         batched=True,
        #         num_proc=args.preprocessing_num_workers,
        #         load_from_cache_file=not args.overwrite_cache,
        #         desc=f"Grouping texts in chunks of {block_size}",
        #     )
        lm_datasets = tokenized_datasets.map(
            truncate_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Truncating inputs",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    custom_collate_fn = collate_fn_builder(
        tokenizer.unk_token_id
    )  # use "<|endoftext|>" as padding for attn_matrix
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        # collate_fn=default_data_collator,
        collate_fn=custom_collate_fn,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        # collate_fn=default_data_collator,
        collate_fn=custom_collate_fn,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties. # noqa
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed. # noqa
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        experiment_config["attn_loss_layers"] = torch.tensor(
            experiment_config["attn_loss_layers"]
        )
        experiment_config["attn_loss_heads"] = torch.tensor(
            experiment_config["attn_loss_heads"]
        )
        # accelerator.init_trackers("clm_no_trainer", experiment_config)
        accelerator.init_trackers("sllm_trainer", experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"  # noqa
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    compute_attn_loss = compute_attn_loss_builder(args.attn_loss_reduction)
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
            total_nwp_loss = 0
            total_attn_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint # noqa
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                # outputs = model(**batch)
                outputs = model(
                    batch["input_ids"],
                    labels=batch["labels"],
                    output_attentions=True,
                )
                # loss = outputs.loss
                nwp_loss = outputs.loss
                attn_loss = compute_attn_loss(
                    outputs.attentions,
                    batch["attn_matrix"],
                    batch["row_word_token_membership_mask"],
                    batch["col_word_token_membership_mask"],
                    args.attn_loss_layers,
                    args.attn_loss_heads,
                )
                loss = nwp_loss + args.attn_loss_weight * attn_loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                    total_nwp_loss += nwp_loss.detach().float()
                    total_attn_loss += attn_loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes # noqa
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        nwp_losses = []
        attn_losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                # outputs = model(**batch)
                outputs = model(
                    batch["input_ids"],
                    labels=batch["labels"],
                    output_attentions=True,
                )
                attn_loss = compute_attn_loss(
                    outputs.attentions,
                    batch["attn_matrix"],
                    batch["row_word_token_membership_mask"],
                    batch["col_word_token_membership_mask"],
                    args.attn_loss_layers,
                    args.attn_loss_heads,
                )
            # loss = outputs.loss
            nwp_loss = outputs.loss
            loss = nwp_loss + args.attn_loss_weight * attn_loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(args.per_device_eval_batch_size)
                )
            )
            nwp_losses.append(
                accelerator.gather_for_metrics(
                    nwp_loss.repeat(args.per_device_eval_batch_size)
                )
            )
            attn_losses.append(
                accelerator.gather_for_metrics(
                    attn_loss.repeat(args.per_device_eval_batch_size)
                )
            )

        losses = torch.cat(losses)
        nwp_losses = torch.cat(nwp_losses)
        attn_losses = torch.cat(attn_losses)
        try:
            eval_loss = torch.mean(losses)
            eval_nwp_loss = torch.mean(nwp_losses)
            eval_attn_loss = torch.mean(attn_losses)
            perplexity = math.exp(eval_nwp_loss)
            # perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        # logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
        logger.info(
            f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss} eval_nwp_loss: {eval_nwp_loss} eval_attn_loss: {eval_attn_loss}"  # noqa
        )

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "eval_nwp_loss": eval_nwp_loss,
                    "eval_attn_loss": eval_attn_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "train_nwp_loss": total_nwp_loss.item() / len(train_dataloader),
                    "train_attn_loss": total_attn_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        # if args.push_to_hub and epoch < args.num_train_epochs - 1:
        #     accelerator.wait_for_everyone()
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_model.save_pretrained(
        #         args.output_dir,
        #         is_main_process=accelerator.is_main_process,
        #         save_function=accelerator.save,
        #     )
        #     if accelerator.is_main_process:
        #         tokenizer.save_pretrained(args.output_dir)
        #         repo.push_to_hub(
        #             commit_message=f"Training in progress epoch {epoch}",
        #             blocking=False,
        #             auto_lfs_prune=True,
        #         )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            # tokenizer.save_pretrained(args.output_dir)
            # if args.push_to_hub:
            #     repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True) # noqa

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
