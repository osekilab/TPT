import argparse
import logging
import os
import random
import re
from typing import IO, Callable, Dict, List, Tuple

import spacy
import torch
from accelerate.utils import set_seed
from datasets import load_dataset
from nltk.tokenize import TreebankWordDetokenizer, TreebankWordTokenizer
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


detokenizer = TreebankWordDetokenizer()
nlp = spacy.load("en_core_web_sm")


BOS = "<s>"
EOS = "<eos>"

LRB = "-LRB-"
RRB = "-RRB-"
LCB = "-LCB-"
RCB = "-RCB-"
LSB = "-LSB-"
RSB = "-RSB-"


def get_structure(model_path: str) -> str:
    m = re.match(r"(.*?)\.structure=([^.]+)\.", model_path)
    if m is None:
        raise ValueError("Invalid model_path")
    structure = m.group(2)
    if structure not in {"constituency", "dependency"}:
        raise ValueError("Invalid structure")
    return structure


def _get_gpt2tokenizer_constituency() -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"bos_token": BOS, "eos_token": EOS})
    tokenizer.add_tokens([LRB, RRB, LCB, RCB, LSB, RSB])
    return tokenizer


def _get_gpt2tokenizer_dependency() -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"bos_token": BOS, "eos_token": EOS})
    return tokenizer


def get_gpt2tokenizer(structure: str) -> GPT2Tokenizer:
    if structure == "constituency":
        return _get_gpt2tokenizer_constituency()
    elif structure == "dependency":
        return _get_gpt2tokenizer_dependency()
    else:
        raise ValueError("Invalid structure")


def tokenize_function_builder(
    structure: str,
    treebank_tokenized: bool,
    text_column_name: str,
    gpt2tokenizer: GPT2Tokenizer,
    treebank_tokenizer: TreebankWordTokenizer,
    bos_id: int,
    eos_id: int,
) -> Callable[[Dict[str, List[str]]], Dict[str, List[int]]]:
    def tokenize_function(examples):
        tokenized_examples = []

        for text in examples[text_column_name]:
            if structure == "constituency":
                if treebank_tokenized:
                    tokenized_text = text
                else:
                    tokenized_text = " ".join(treebank_tokenizer.tokenize(text))
            elif structure == "dependency":
                if treebank_tokenized:
                    text = detokenizer.detokenize(text.split())
                tokenized_text = " ".join([token.text for token in nlp.make_doc(text)])

            input_ids = gpt2tokenizer.encode(tokenized_text)
            tokenized_examples.append([bos_id] + input_ids + [eos_id])

        return {"input_ids": tokenized_examples}

    return tokenize_function


def unified_tokenizer_builder(
    structure: str,
    treebank_tokenized: bool,
    gpt2tokenizer: GPT2Tokenizer,
    treebank_tokenizer: TreebankWordTokenizer,
) -> Callable[[str], List[str]]:
    def unified_tokenizer(text: str) -> List[str]:
        if structure == "constituency":
            if treebank_tokenized:
                tokenized_text = text
            else:
                tokenized_text = " ".join(treebank_tokenizer.tokenize(text))
        elif structure == "dependency":
            if treebank_tokenized:
                text = detokenizer.detokenize(text.split())
            tokenized_text = " ".join([token.text for token in nlp.make_doc(text)])

        return gpt2tokenizer.tokenize(tokenized_text)

    return unified_tokenizer


def collate_fn_builder(pad_id: int = 50256):
    def collate_fn(batch: Dict[str, torch.Tensor]):
        input_ids, text = [], []
        for item in batch:
            input_ids.append(torch.tensor(item["input_ids"]))
            text.append(item["text"])

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )

        return {"input_ids": input_ids_padded, "text": text}

    return collate_fn


def _convert_logits2surps(
    logits: torch.Tensor,  # (batch_size, seq_len, vocab_size)
    input_ids: torch.Tensor,  # (batch_size, seq_len)
) -> torch.Tensor:  # (batch_size, seq_len-2); -2 for <bos> and <eos>
    softmax = torch.nn.functional.softmax(logits, dim=-1)
    surps = -torch.log2(softmax)[..., :-2, :].contiguous()
    shifted_input_ids = input_ids[..., 1:-1].contiguous()

    row_indices = torch.arange(shifted_input_ids.shape[0]).unsqueeze(1)
    col_indices = torch.arange(shifted_input_ids.shape[1]).unsqueeze(0)
    surps = surps[
        row_indices, col_indices, shifted_input_ids
    ]  # (batch_size, seq_len-2)
    return surps


def _calc_and_save_in_sent_surps(
    token_strs: List[str],
    surps: torch.Tensor,
    sentence_id: int,
    total_surps: float,
    total_n_tokens: int,
    f: IO[str],
) -> Tuple[int, float, int]:
    def write_word_surp(
        sentence_id: int,
        word_id: int,
        orig_word: str,
        in_word_tokens: List[str],
        in_word_surps: List[float],
    ) -> None:
        modified_word = " ".join(in_word_tokens)
        surp = sum(in_word_surps)
        in_word_surps_str = " ".join([str(s) for s in in_word_surps])
        f.write(
            f"{sentence_id}\t{word_id}\t{orig_word}\t{modified_word}\t{surp}\t{in_word_surps_str}\n"  # noqa
        )

    # words = text.split()
    word_id_pointer = 0
    in_word_surps = []
    in_word_tokens = []
    for token_str, surp in zip(token_strs, surps):
        if token_str.startswith("Ġ") and in_word_surps:
            write_word_surp(
                sentence_id,
                word_id_pointer,
                "".join(in_word_tokens),
                in_word_tokens,
                in_word_surps,
            )
            word_id_pointer += 1
            in_word_surps = []
            in_word_tokens = []
        in_word_tokens.append(token_str.replace("Ġ", ""))
        in_word_surps.append(surp.item())
        total_surps += surp.item()
        total_n_tokens += 1

    if in_word_surps:
        write_word_surp(
            sentence_id,
            word_id_pointer,
            "".join(in_word_tokens),
            in_word_tokens,
            in_word_surps,
        )
    sentence_id += 1
    return sentence_id, total_surps, total_n_tokens


def calc_and_save_in_batch_surps(
    logits: torch.Tensor,  # (batch_size, seq_len, vocab_size)
    input_ids: torch.Tensor,  # (batch_size, seq_len)
    texts: List[str],
    sentence_id: int,
    total_surps: float,
    total_n_tokens: int,
    unified_tokenizer: Callable[[str], List[str]],
    f: IO[str],
) -> Tuple[float, int]:
    batch_surps = _convert_logits2surps(logits, input_ids)
    batch_token_strs = [unified_tokenizer(text) for text in texts]

    for token_strs, surps in zip(batch_token_strs, batch_surps):
        sentence_id, total_surps, total_n_tokens = _calc_and_save_in_sent_surps(
            token_strs,
            surps,
            sentence_id,
            total_surps,
            total_n_tokens,
            f,
        )
    return sentence_id, total_surps, total_n_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--treebank_tokenized", action="store_true")
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
    args = parser.parse_args()

    device = "cpu" if args.use_cpu else f"cuda:{args.gpu_id}"

    logger.warning(f"device: {device}, 16-bits inference: {args.fp16}")

    if args.seed is not None:
        set_seed(args.seed)

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    data_files = {}
    data_files["test"] = args.test_file
    raw_datasets = load_dataset("text", data_files=data_files)

    treebank_tokenizer = TreebankWordTokenizer()

    structure = get_structure(args.model_path)
    gpt2tokenizer = get_gpt2tokenizer(structure)

    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    if args.fp16:
        model.half()

    column_names = raw_datasets["test"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    tokenize_function = tokenize_function_builder(
        structure,
        args.treebank_tokenized,
        text_column_name,
        gpt2tokenizer,
        treebank_tokenizer,
        gpt2tokenizer.bos_token_id,
        gpt2tokenizer.eos_token_id,
    )
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    if args.block_size is None:
        block_size = min(1024, model.config.n_positions)

    def truncate_function(examples):
        return {"input_ids": [t[:block_size] for t in examples["input_ids"]]}

    lm_datasets = tokenized_datasets.map(
        truncate_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Truncating inputs",
    )

    test_dataset = lm_datasets["test"]

    for index in random.sample(range(len(test_dataset)), 3):
        logger.info(f"Sample {index} of the text set: {test_dataset[index]}.")

    custom_collate_fn = collate_fn_builder(pad_id=gpt2tokenizer.unk_token_id)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
    )

    unified_tokenizer = unified_tokenizer_builder(
        structure,
        args.treebank_tokenized,
        gpt2tokenizer,
        treebank_tokenizer,
    )
    model.eval()
    with torch.no_grad() and open(args.output_path, "w") as f:
        f.write("sentence_id\tword_id\torig_word\tmodified_word\tsurp\tin_word_surps\n")
        sentence_id = 0
        total_surps = 0.0
        total_n_tokens = 0
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            sentence_id, total_surps, total_n_tokens = calc_and_save_in_batch_surps(
                logits,
                input_ids,
                batch["text"],
                sentence_id,
                total_surps,
                total_n_tokens,
                unified_tokenizer,
                f,
            )
        perplexity = torch.exp2(torch.tensor(total_surps / total_n_tokens))
        f.write("========================================\n")
        f.write(f"perplexity: {perplexity}")


if __name__ == "__main__":
    main()
