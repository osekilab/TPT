import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import nltk
import numpy as np
import spacy
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import GPT2Tokenizer

from utils import configure_logging, logging

nlp = spacy.load("en_core_web_sm")

BOS = "<s>"
EOS = "<eos>"

TokenOrStr = Union[spacy.tokens.token.Token, str]

detokenizer = nltk.tokenize.TreebankWordDetokenizer()


def find_existing_preprocessed_files(output_path: str) -> List[str]:
    directory_path = os.path.dirname(output_path)
    file_prefix = os.path.basename(output_path).split(".", 1)[0]
    pattern = (
        f"{file_prefix}.sequential=False." + "random=False." "convert_method=*.jsonl"
    )
    full_path_pattern = os.path.join(directory_path, pattern)
    matching_files = glob.glob(full_path_pattern)
    return matching_files


def get_gpt2tokenizer() -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"bos_token": BOS, "eos_token": EOS})
    return tokenizer


def compute_n_edge_distribution(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    all_n_edges = []
    with open(file_path, "r") as infile:
        for line in tqdm(infile, position=0, leave=True):
            data = json.loads(line.strip())
            n_edge_matrix = np.array(data["n_edge_matrix"])
            all_n_edges.extend(n_edge_matrix[n_edge_matrix != 0].tolist())
    all_n_edges = np.array(all_n_edges) / 2  # all elems are counted twice

    elements, counts = np.unique(all_n_edges, return_counts=True)
    probs = counts / counts.sum()
    return elements, probs


def _compute_dep_path_to_root(
    token: spacy.tokens.token.Token,
) -> List[spacy.tokens.token.Token]:
    path = []
    while token.dep_ != "ROOT":
        path.append(token)
        token = token.head
    path.append(token)
    return path


def _compute_dep_paths(
    doc: spacy.tokens.Doc,
) -> Tuple[
    Dict[spacy.tokens.token.Token, List[spacy.tokens.token.Token]],  # paths
    spacy.tokens.token.Token,  # root
]:
    paths = {}
    root = None
    for token in doc:
        paths[token] = _compute_dep_path_to_root(token)
        if token.dep_ == "ROOT":
            root = token
    return paths, root


def _compute_shortest_dep_path_len(
    paths: Dict[TokenOrStr, List[TokenOrStr]],
    token1: TokenOrStr,
    token2: TokenOrStr,
) -> int:
    paths1 = set(paths[token1])
    paths2 = set(paths[token2])

    common_ancestors = paths1.intersection(paths2)
    if not common_ancestors:
        return -1

    common_ancestor = min(
        common_ancestors,
        key=lambda ancestor: paths[token1].index(ancestor)
        + paths[token2].index(ancestor),
    )
    path_len = paths[token1].index(common_ancestor) + paths[token2].index(
        common_ancestor
    )
    return path_len


def compute_dep_path_len_matrix_adding_bos_eos(
    sentence: str,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    doc = nlp(sentence)
    paths, root = _compute_dep_paths(doc)

    paths_w_bos_eos = {BOS: [BOS, root], **paths, EOS: [EOS, root]}
    words = list(paths_w_bos_eos.keys())

    path_len_matrix = np.zeros((len(words), len(words)), dtype=np.int32)
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words[i:], start=i):  # Only compute upper triangle
            shortest_dep_path_len = _compute_shortest_dep_path_len(
                paths_w_bos_eos, word1, word2
            )
            if shortest_dep_path_len == -1:
                return None, None
            path_len_matrix[i, j] = shortest_dep_path_len
            path_len_matrix[j, i] = path_len_matrix[i, j]
    word_strs_wo_bos_eos = [word.text for word in paths.keys()]
    return path_len_matrix, word_strs_wo_bos_eos


def _gen_sequential_distance_matrix_from_n_edge_matrix(
    n_edge_matrix: np.ndarray,
) -> np.ndarray:
    n = n_edge_matrix.shape[0]
    indices = np.arange(n)
    return np.abs(indices.reshape(n, 1) - indices)


def _gen_random_distance_matrix_from_n_edge_matrix(
    n_edge_matrix: np.ndarray,
    elements: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    n = n_edge_matrix.shape[0]
    return np.random.choice(elements, size=(n, n), p=probs)


def _extract_lower_triangle(A: np.ndarray) -> np.ndarray:
    lower_triangle = np.tril(A, -1)
    lower_triangle = np.delete(lower_triangle, -1, axis=1)
    lower_triangle = np.delete(lower_triangle, 0, axis=0)
    return lower_triangle


def _nonzero_reciprocal(A: np.ndarray) -> np.ndarray:
    return np.divide(1.0, A, out=np.zeros_like(A, dtype=float), where=A != 0)


def _nonzero_exponential(A: np.ndarray) -> np.ndarray:
    return np.exp(-A, where=A != 0, out=np.zeros_like(A, dtype=float))


def _nonzero_linear(A: np.ndarray, linear_intercept: float) -> np.ndarray:
    return np.where(A != 0, -A + linear_intercept, 0)


def _nonzero_log(A: np.ndarray, log_offset: float) -> np.ndarray:
    return np.where(
        A != 0,
        (np.log(A + log_offset) / np.log(0.5)) - (np.log(log_offset) / np.log(0.5)),
        0,
    )


def _nonzero_sigmoid(A: np.ndarray, sigmoid_offset: float) -> np.ndarray:
    return np.where(A != 0, 1 / (1 + np.exp(0.2 * (A - (sigmoid_offset / 2)))), 0)


def _normalize_rows2one(A: np.ndarray) -> np.ndarray:
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A /= row_sums
    return A


def convert_fn_builder(
    convert_method: str,
    linear_intercept: float = 55,
    log_offset: float = 55,
    sigmoid_offset: float = 55,
) -> Callable[[np.ndarray], np.ndarray]:
    if convert_method == "reciprocal":
        return _nonzero_reciprocal
    elif convert_method == "exponential":
        return _nonzero_exponential
    elif convert_method == "linear":
        return partial(_nonzero_linear, linear_intercept=linear_intercept)
    elif convert_method == "log":
        return partial(_nonzero_log, log_offset=log_offset)
    elif convert_method == "sigmoid":
        return partial(_nonzero_sigmoid, sigmoid_offset=sigmoid_offset)
    else:
        raise NotImplementedError(
            f"Conversion method '{convert_method}' is not implemented."
        )


def convert_n_edge_matrix2attn_matrix(
    n_edge_matrix: np.ndarray,
    sequential: bool,
    random: bool,
    elements: Optional[np.ndarray],
    probs: Optional[np.ndarray],
    convert_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    _n_edge_matrix = n_edge_matrix.copy()
    if sequential:
        _n_edge_matrix = _gen_sequential_distance_matrix_from_n_edge_matrix(
            n_edge_matrix
        )
    elif random:
        _n_edge_matrix = _gen_random_distance_matrix_from_n_edge_matrix(
            n_edge_matrix, elements, probs
        )
    attn_shaped_n_edge_matrix = _extract_lower_triangle(_n_edge_matrix)
    converted_matrix = convert_fn(attn_shaped_n_edge_matrix)
    attn_matrix = _normalize_rows2one(converted_matrix)
    assert np.all(attn_matrix >= 0)
    return attn_matrix


def apply_convert_n_edge_matrix2attn_matrix(
    line: str,
    sequential: bool,
    random: bool,
    elements: Optional[np.ndarray],
    probs: Optional[np.ndarray],
    convert_fn: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, Any]:
    data = json.loads(line.strip())
    data["n_edge_matrix"] = np.array(data["n_edge_matrix"])
    data["attn_matrix"] = convert_n_edge_matrix2attn_matrix(
        data["n_edge_matrix"],
        sequential,
        random,
        elements,
        probs,
        convert_fn,
    )
    return data


def _group_token_indices_in_words(
    token_strs: List[str],
    bos_eos_tokens: Set[str],
) -> List[List[int]]:
    grouped_indices = []
    current_group = []

    for i, token in enumerate(token_strs):
        if token.startswith("Ġ") or i == 1 or token in bos_eos_tokens:
            # First token of a sentence, or
            # a token immediately after BOS, or
            # a special token
            if current_group:
                grouped_indices.append(current_group)
            current_group = [i]
        else:
            current_group.append(i)

    if current_group:
        grouped_indices.append(current_group)

    return grouped_indices


def create_word_token_membership_mask(
    token_strs: List[str], bos_eos_tokens: Set[str]
) -> np.ndarray:
    wordwise_token_indices = _group_token_indices_in_words(token_strs, bos_eos_tokens)
    token_indices = np.arange(len(token_strs))
    mask = np.array(
        [
            np.isin(token_indices, _wordwise_token_indices)
            for _wordwise_token_indices in wordwise_token_indices
        ]
    )  # (n_words, n_tokens)
    return mask


# def convert_word_token_membership_mask2reduce_matrices(
#     word_token_membership_mask: np.ndarray,
# ) -> np.ndarray:
#     row_reduce_matrix = word_token_membership_mask[
#         :-1, :-1
#     ].copy()  # remove EOS; attended tokens
#     col_reduce_matrix = word_token_membership_mask[
#         1:, 1:
#     ].copy()  # remove BOS; predicted tokens

#     return row_reduce_matrix, col_reduce_matrix


def preprocess(
    line: str,
    gpt2tokenizer: GPT2Tokenizer,
    min_n_terminals: int,
    max_n_terminals: int,
    sequential: bool,
    random: bool,
    elements: Optional[np.ndarray],
    probs: Optional[np.ndarray],
    convert_fn: Callable[[np.ndarray], np.ndarray],
) -> Optional[Tuple[str, List[str], str, List[str], List[int], np.ndarray, np.ndarray]]:
    tree = nltk.Tree.fromstring(line.strip())
    terminal_strs = tree.leaves()
    n_terminals = len(terminal_strs)
    if n_terminals < min_n_terminals or max_n_terminals < n_terminals:
        return None

    treebank_detokenized_sentence = detokenizer.detokenize(
        terminal_strs, convert_parentheses=True
    )  # noqa
    (
        dep_path_len_matrix,
        word_strs_wo_bos_eos,
    ) = compute_dep_path_len_matrix_adding_bos_eos(
        treebank_detokenized_sentence
    )  # word_strs is tokenized with detokenizer -> spacy
    if dep_path_len_matrix is None:
        return None

    token_strs = [BOS] + gpt2tokenizer.tokenize(" ".join(word_strs_wo_bos_eos)) + [EOS]
    token_ids = gpt2tokenizer.convert_tokens_to_ids(token_strs)
    subword_sentence = " ".join(token_strs)

    attn_matrix = convert_n_edge_matrix2attn_matrix(
        dep_path_len_matrix,
        sequential,
        random,
        elements,
        probs,
        convert_fn,
    )

    word_token_membership_mask = create_word_token_membership_mask(
        token_strs,
        bos_eos_tokens={BOS, EOS},
    )
    # move to the `train.py` because I/O is slow
    # (
    #     row_reduce_matrix,
    #     col_reduce_matrix,
    # ) = convert_word_token_membership_mask2reduce_matrices(word_token_membership_mask)

    return (
        treebank_detokenized_sentence,
        word_strs_wo_bos_eos,
        subword_sentence,
        token_strs,
        token_ids,
        dep_path_len_matrix,
        attn_matrix,
        word_token_membership_mask,
        # row_reduce_matrix,
        # col_reduce_matrix,
    )


def preprocess_file(
    input_path: str,
    output_path: str,
    gpt2tokenizer: GPT2Tokenizer,
    min_n_terminals: int,
    max_n_terminals: int,
    n_processes: int,
    sequential: bool,
    random: bool,
    elements: Optional[np.ndarray],
    probs: Optional[np.ndarray],
    convert_fn: Callable[[np.ndarray], np.ndarray],
) -> Tuple[int, int]:
    if random:
        assert elements is not None
        assert probs is not None
    filtered_count = 0
    total_count = 0
    chunk_size = n_processes * 100
    with open(input_path, "r") as infile, open(output_path, "w+") as outfile:
        chunk = []
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            for line in tqdm(infile, position=0, leave=True):
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    futures = {
                        executor.submit(
                            preprocess,
                            line,
                            gpt2tokenizer,
                            min_n_terminals,
                            max_n_terminals,
                            sequential,
                            random,
                            elements,
                            probs,
                            convert_fn,
                        ): index
                        for index, line in enumerate(chunk)
                    }
                    results = []
                    for future in as_completed(futures):
                        index = futures[future]
                        processed = future.result()
                        total_count += 1
                        if processed is None:
                            filtered_count += 1
                            continue
                        results.append((index, processed))
                    results.sort(key=lambda x: x[0])
                    for _, processed in results:
                        (
                            treebank_detokenized_sentence,
                            word_strs_wo_bos_eos,
                            subword_sentence,
                            token_strs,
                            token_ids,
                            dep_path_len_matrix,
                            attn_matrix,
                            word_token_membership_mask,
                            # row_reduce_matrix,
                            # col_reduce_matrix,
                        ) = processed

                        assert (
                            len(word_strs_wo_bos_eos) + 2
                            == dep_path_len_matrix.shape[0]
                            == dep_path_len_matrix.shape[1]
                            == attn_matrix.shape[0] + 1
                            == attn_matrix.shape[1] + 1
                            == word_token_membership_mask.shape[0]
                        )
                        assert (
                            len(token_strs)
                            == len(token_ids)
                            == word_token_membership_mask.shape[1]
                        )

                        data = {
                            "treebank_detokenized_sentence": treebank_detokenized_sentence,  # noqa
                            "word_strs_wo_bos_eos": word_strs_wo_bos_eos,
                            "subword_sentence": subword_sentence,
                            "token_strs": token_strs,
                            "token_ids": token_ids,
                            "n_edge_matrix": dep_path_len_matrix.tolist(),
                            "attn_matrix": attn_matrix.tolist(),
                            "word_token_membership_mask": word_token_membership_mask.tolist(),  # noqa
                            # "row_reduce_matrix": row_reduce_matrix.tolist(),
                            # "col_reduce_matrix": col_reduce_matrix.tolist(),
                        }
                        json.dump(data, outfile)
                        outfile.write("\n")
                    chunk = []
            if chunk:
                futures = {
                    executor.submit(
                        preprocess,
                        line,
                        gpt2tokenizer,
                        min_n_terminals,
                        max_n_terminals,
                        sequential,
                        random,
                        elements,
                        probs,
                        convert_fn,
                    ): index
                    for index, line in enumerate(chunk)
                }
                results = []
                for future in as_completed(futures):
                    index = futures[future]
                    processed = future.result()
                    total_count += 1
                    if processed is None:
                        filtered_count += 1
                        continue
                    results.append((index, processed))
                results.sort(key=lambda x: x[0])
                for _, processed in results:
                    (
                        treebank_detokenized_sentence,
                        word_strs_wo_bos_eos,
                        subword_sentence,
                        token_strs,
                        token_ids,
                        dep_path_len_matrix,
                        attn_matrix,
                        word_token_membership_mask,
                        # row_reduce_matrix,
                        # col_reduce_matrix,
                    ) = processed

                    assert (
                        len(word_strs_wo_bos_eos) + 2
                        == dep_path_len_matrix.shape[0]
                        == dep_path_len_matrix.shape[1]
                        == attn_matrix.shape[0] + 1
                        == attn_matrix.shape[1] + 1
                        == word_token_membership_mask.shape[0]
                    )
                    assert (
                        len(token_strs)
                        == len(token_ids)
                        == word_token_membership_mask.shape[1]
                    )

                    data = {
                        "treebank_detokenized_sentence": treebank_detokenized_sentence,
                        "word_strs_wo_bos_eos": word_strs_wo_bos_eos,
                        "subword_sentence": subword_sentence,
                        "token_strs": token_strs,
                        "token_ids": token_ids,
                        "n_edge_matrix": dep_path_len_matrix.tolist(),
                        "attn_matrix": attn_matrix.tolist(),
                        "word_token_membership_mask": word_token_membership_mask.tolist(),  # noqa
                        # "row_reduce_matrix": row_reduce_matrix.tolist(),
                        # "col_reduce_matrix": col_reduce_matrix.tolist(),
                    }
                    json.dump(data, outfile)
                    outfile.write("\n")
    return total_count, filtered_count


def preprocess_with_existing_file(
    ref_path: str,
    output_path: str,
    n_processes: int,
    sequential: bool,
    random: bool,
    elements: Optional[np.ndarray],
    probs: Optional[np.ndarray],
    convert_fn: Callable[[np.ndarray], np.ndarray],
) -> Tuple[int]:
    total_count = 0
    chunk_size = n_processes * 100
    with open(ref_path, "r") as infile, open(output_path, "+w") as outfile:
        chunk = []
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            for line in tqdm(infile, position=0, leave=True):
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    futures = {
                        executor.submit(
                            apply_convert_n_edge_matrix2attn_matrix,
                            line,
                            sequential,
                            random,
                            elements,
                            probs,
                            convert_fn,
                        ): index
                        for index, line in enumerate(chunk)
                    }
                    results = []
                    for future in as_completed(futures):
                        index = futures[future]
                        data = future.result()
                        total_count += 1
                        assert (
                            data["n_edge_matrix"].shape[0]
                            == data["attn_matrix"].shape[0] + 1
                            == data["attn_matrix"].shape[1] + 1
                        )
                        data["n_edge_matrix"] = data["n_edge_matrix"].tolist()
                        data["attn_matrix"] = data["attn_matrix"].tolist()
                        results.append((index, data))
                    results.sort(key=lambda x: x[0])
                    for _, data in results:
                        json.dump(data, outfile)
                        outfile.write("\n")
                    chunk = []
            if chunk:
                futures = {
                    executor.submit(
                        apply_convert_n_edge_matrix2attn_matrix,
                        line,
                        sequential,
                        random,
                        elements,
                        probs,
                        convert_fn,
                    ): index
                    for index, line in enumerate(chunk)
                }
                results = []
                for future in as_completed(futures):
                    index = futures[future]
                    data = future.result()
                    total_count += 1
                    assert (
                        data["n_edge_matrix"].shape[0]
                        == data["attn_matrix"].shape[0] + 1
                        == data["attn_matrix"].shape[1] + 1
                    )
                    data["n_edge_matrix"] = data["n_edge_matrix"].tolist()
                    data["attn_matrix"] = data["attn_matrix"].tolist()
                    results.append((index, data))
                results.sort(key=lambda x: x[0])
                for _, data in results:
                    json.dump(data, outfile)
                    outfile.write("\n")
    return total_count


def main():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument("--train_file_path", type=str, required=True)
    parser.add_argument("--val_file_path", type=str, required=True)
    parser.add_argument("--test_file_path", type=str, required=True)
    # filter
    parser.add_argument("--max_n_terminals", type=int, default=10000)
    parser.add_argument("--min_n_terminals", type=int, default=0)
    # edge
    parser.add_argument(
        "--convert_method",
        type=str,
        default="exponential",
        choices=[
            "reciprocal",
            "exponential",
            "linear",
            "log",
            "sigmoid",
        ],
    )
    parser.add_argument(
        "--linear_intercept",
        type=float,
        default=55,
        help=(
            "The intercept for the linear function. Set this above the "
            "maximum 'n_edge_matrix' value to prevent negative results "
            "after linear transformation."
        ),
    )
    parser.add_argument(
        "--log_offset",
        type=float,
        default=55,
        help=(
            "The offset for the log function. Set this above the maximum "
            "'n_edge_matrix' value to prevent negative results after "
            "log transformation."
        ),
    )
    parser.add_argument(
        "--sigmoid_offset",
        type=float,
        default=55,
        help=(
            "The center for the sigmoid function. Set this above the "
            "maximum 'n_edge_matrix' value to prevent negative results "
            "after sigmoid transformation."
        ),
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Use sequential distance."
    )
    parser.add_argument("--random", action="store_true", help="Use random distance.")
    # output
    parser.add_argument(
        "--output_dir", type=str, default="data/preprocessed_v2/dependency/"
    )
    parser.add_argument("--overwrite", action="store_true")
    # process
    parser.add_argument("--n_processes", type=int, default=-1)
    # seed
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    configure_logging(
        args.output_dir
        + f"preprocess.sequential={args.sequential}.random={args.random}.convert_method={args.convert_method}.log"  # noqa
    )
    logger = logging.getLogger(__name__)

    logger.info("Arguments:")
    logger.info(args)

    logger.info("Start preprocess.py")

    if args.n_processes == -1:
        args.n_processes = os.cpu_count()

    gpt2tokenizer = get_gpt2tokenizer()

    convert_fn = convert_fn_builder(
        args.convert_method,
        linear_intercept=args.linear_intercept,
        log_offset=args.log_offset,
        sigmoid_offset=args.sigmoid_offset,
    )

    train_output_path = (
        args.output_dir
        + f"train.sequential={args.sequential}.random={args.random}.convert_method={args.convert_method}.jsonl"  # noqa
    )
    if os.path.exists(train_output_path) and not args.overwrite:
        logger.info("Train file already exists. Skip preprocessing train file.")
    else:
        existing_preprocessed_train_files = find_existing_preprocessed_files(
            train_output_path
        )
        if existing_preprocessed_train_files:
            logger.info(
                "Preprocess with `n_edge_matrix` from existing preprocessed train file"
            )
            if args.random:
                elements, probs = compute_n_edge_distribution(
                    existing_preprocessed_train_files[0]
                )
            else:
                elements, probs = None, None
            train_total_count = preprocess_with_existing_file(
                existing_preprocessed_train_files[0],
                train_output_path,
                args.n_processes,
                args.sequential,
                args.random,
                elements,
                probs,
                convert_fn,
            )
            logger.info("Number of train sentences: {}".format(train_total_count))
        else:
            if args.random:
                raise ValueError(
                    "Random distance matrix is not available without existing preprocessed train file."  # noqa
                )
            elements, probs = None, None
            logger.info("Preprocess train file")
            train_total_count, train_filtered_count = preprocess_file(
                args.train_file_path,
                train_output_path,
                gpt2tokenizer,
                args.min_n_terminals,
                args.max_n_terminals,
                args.n_processes,
                args.sequential,
                args.random,
                elements,
                probs,
                convert_fn,
            )
            logger.info(
                "Number of train sentences: {} ({} filtered)".format(
                    train_total_count, train_filtered_count
                )
            )

    val_output_path = (
        args.output_dir
        + f"val.sequential={args.sequential}.random={args.random}.convert_method={args.convert_method}.jsonl"  # noqa
    )
    if os.path.exists(val_output_path) and not args.overwrite:
        logger.info("Val file already exists. Skip preprocessing val file.")
    else:
        existing_preprocessed_val_files = find_existing_preprocessed_files(
            val_output_path
        )
        if existing_preprocessed_val_files:
            logger.info(
                "Preprocess with `n_edge_matrix` from existing preprocessed val file"
            )
            val_total_count = preprocess_with_existing_file(
                existing_preprocessed_val_files[0],
                val_output_path,
                args.n_processes,
                args.sequential,
                args.random,
                elements,
                probs,
                convert_fn,
            )
            logger.info("Number of val sentences: {}".format(val_total_count))
        else:
            logger.info("Preprocess val file")
            val_total_count, val_filtered_count = preprocess_file(
                args.val_file_path,
                val_output_path,
                gpt2tokenizer,
                args.min_n_terminals,
                args.max_n_terminals,
                args.n_processes,
                args.sequential,
                args.random,
                elements,
                probs,
                convert_fn,
            )
            logger.info(
                "Number of val sentences: {} ({} filtered)".format(
                    val_total_count, val_filtered_count
                )
            )

    test_output_path = (
        args.output_dir
        + f"test.sequential={args.sequential}.random={args.random}.convert_method={args.convert_method}.jsonl"  # noqa
    )
    if os.path.exists(test_output_path) and not args.overwrite:
        logger.info("Test file already exists. Skip preprocessing test file.")
    else:
        existing_preprocessed_test_files = find_existing_preprocessed_files(
            test_output_path
        )
        if existing_preprocessed_test_files:
            logger.info(
                "Preprocess with `n_edge_matrix` from existing preprocessed test file"
            )
            test_total_count = preprocess_with_existing_file(
                existing_preprocessed_test_files[0],
                test_output_path,
                args.n_processes,
                args.sequential,
                args.random,
                elements,
                probs,
                convert_fn,
            )
            logger.info("Number of test sentences: {}".format(test_total_count))
        else:
            logger.info("Preprocess test file")
            test_total_count, test_filtered_count = preprocess_file(
                args.test_file_path,
                test_output_path,
                gpt2tokenizer,
                args.min_n_terminals,
                args.max_n_terminals,
                args.n_processes,
                args.sequential,
                args.random,
                elements,
                probs,
                convert_fn,
            )
            logger.info(
                "Number of test sentences: {} ({} filtered)".format(
                    test_total_count, test_filtered_count
                )
            )
    logger.info("Finish preprocess.py")


if __name__ == "__main__":
    main()
