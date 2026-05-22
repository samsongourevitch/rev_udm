import functools
import glob
import itertools
import json
import math
import os
import re
import shutil
import tarfile
import typing
import urllib
import zipfile
from typing import Optional

import datasets
import fsspec
import numpy as np
import requests
import tokenizers
import torch
import transformers

import utils

LOGGER = utils.get_logger(__name__)


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x


def lm1b_detokenizer(x):
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r"\" ([^\"]+) \"", r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return "\n" + text.strip()


def scientific_papers_detokenizer(x):
    x = wt_detokenizer(x)
    x = lm1b_detokenizer(x)
    return x


class SyntheticTokenizer(transformers.PreTrainedTokenizer):

    def __init__(
        self,
        vocab_size,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token=None,
        cls_token=None,
        pad_token=None,
        mask_token=None,
        unk_token=None,
        **kwargs,
    ):

        self.tokens = []

        for i in range(vocab_size - 2):
            # appending space for readability
            self.tokens.append(str(i) + " ")

        self._vocab_str_to_int = {
            "[BOS]": vocab_size - 2,
            "[EOS]": vocab_size - 1,
            **{ch: i for i, ch in enumerate(self.tokens)},
        }

        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


def _generate_synthetic_data(dataset_size, seq_len, vocab_size):
    dataset = np.zeros((dataset_size, seq_len), dtype=int)
    # tokens representing sequence boundary
    dataset[:, 0] = vocab_size - 2  # bos
    dataset[:, -1] = vocab_size - 1  # eos

    for i in range(dataset_size):
        # sample from 0, 1, ..., vocab_size - 3
        temp = np.random.randint(vocab_size - 2)
        for j in reversed(range(1, seq_len - 1)):
            dataset[i, j] = temp
            if temp != 0:
                temp = temp // 4
            else:
                temp = np.random.randint(vocab_size - 2)

    return dataset


def generate_synthetic_dataset(
    train_dataset_size, validation_dataset_size, seq_len, vocab_size
):
    np.random.seed(42)
    train_data = torch.from_numpy(
        _generate_synthetic_data(train_dataset_size, seq_len, vocab_size)
    )
    train_dataset = datasets.Dataset.from_dict(
        {
            "input_ids": train_data,
            "attention_mask": torch.ones_like(train_data),
        }
    )
    train_dataset.set_format(type="torch")

    np.random.seed(41)
    validation_data = torch.from_numpy(
        _generate_synthetic_data(validation_dataset_size, seq_len, vocab_size)
    )
    validation_dataset = datasets.Dataset.from_dict(
        {
            "input_ids": validation_data,
            "attention_mask": torch.ones_like(validation_data),
        }
    )
    validation_dataset.set_format(type="torch")

    return {
        "train": train_dataset,
        "validation": validation_dataset,
    }


class Text8Tokenizer(transformers.PreTrainedTokenizer):
    def __init__(
        self,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        self.characters = list("abcdefghijklmnopqrstuvwxyz ")
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[MASK]": 4,
            "[PAD]": 5,
            "[RESERVED]": 6,
            "[UNK]": 7,
            **{ch: i + 8 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset


SCIENTIFIC_PAPERS_URLS = {
    "arxiv": "https://s3.amazonaws.com/datasets.huggingface.co/scientific_papers/1.1.1/arxiv-dataset.zip",
    "pubmed": "https://s3.amazonaws.com/datasets.huggingface.co/scientific_papers/1.1.1/pubmed-dataset.zip",
}

SCIENTIFIC_PAPERS_SPLIT_FILES = {
    "train": "train.txt",
    "validation": "val.txt",
    "test": "test.txt",
}

PTB_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"
PTB_SPLIT_FILES = {
    "train": "ptb.train.txt",
    "validation": "ptb.valid.txt",
    "test": "ptb.test.txt",
}

LM1B_URL = (
    "http://www.statmt.org/lm-benchmark/"
    "1-billion-word-language-modeling-benchmark-r13output.tar.gz"
)
LM1B_TOP_LEVEL_DIR = "1-billion-word-language-modeling-benchmark-r13output"
LM1B_TRAIN_DIR = os.path.join(
    LM1B_TOP_LEVEL_DIR, "training-monolingual.tokenized.shuffled"
)
LM1B_HELDOUT_DIR = os.path.join(
    LM1B_TOP_LEVEL_DIR, "heldout-monolingual.tokenized.shuffled"
)


def get_scientific_papers_dataset(cache_dir, subset, split, num_proc=None):
    if subset not in SCIENTIFIC_PAPERS_URLS:
        raise ValueError(f"Unsupported scientific_papers subset: {subset}")
    if split not in SCIENTIFIC_PAPERS_SPLIT_FILES:
        raise ValueError(f"Unsupported scientific_papers split: {split}")

    raw_cache_dir = os.path.join(cache_dir, "scientific_papers", subset, "raw_data")
    utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)

    zip_filename = os.path.basename(SCIENTIFIC_PAPERS_URLS[subset])
    zip_path = os.path.join(raw_cache_dir, zip_filename)
    extracted_dir = os.path.join(raw_cache_dir, f"{subset}-dataset")
    split_path = os.path.join(extracted_dir, SCIENTIFIC_PAPERS_SPLIT_FILES[split])

    if not utils.fsspec_exists(split_path):
        if not utils.fsspec_exists(zip_path):
            LOGGER.info(
                "Downloading scientific_papers %s from %s",
                subset,
                SCIENTIFIC_PAPERS_URLS[subset],
            )
            with (
                urllib.request.urlopen(SCIENTIFIC_PAPERS_URLS[subset]) as in_stream,
                open(zip_path, "wb") as out_file,
            ):
                shutil.copyfileobj(in_stream, out_file)

        LOGGER.info("Extracting scientific_papers %s from %s", subset, zip_path)
        with fsspec.open(zip_path, "rb") as file_obj:
            with zipfile.ZipFile(file_obj) as archive:
                archive.extractall(raw_cache_dir)

    dataset = datasets.load_dataset(
        "json",
        data_files={split: split_path},
        split=split,
        cache_dir=cache_dir,
    )

    def normalize_record(example):
        summary = "\n".join(example["abstract_text"])
        summary = summary.replace("<S>", "").replace("</S>", "")
        return {
            "article": "\n".join(example["article_text"]),
            "abstract": summary,
            "section_names": "\n".join(example["section_names"]),
        }

    return dataset.map(
        normalize_record,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
        desc=f"Normalizing scientific_papers/{subset} {split}",
    )


def get_ptb_dataset(cache_dir, split):
    if split not in PTB_SPLIT_FILES:
        raise ValueError(f"Unsupported PTB split: {split}")

    raw_cache_dir = os.path.join(cache_dir, "ptb_text_only", "raw_data")
    utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)

    filename = PTB_SPLIT_FILES[split]
    file_path = os.path.join(raw_cache_dir, filename)
    if not utils.fsspec_exists(file_path):
        url = f"{PTB_URL}{filename}"
        LOGGER.info("Downloading PTB %s from %s", split, url)
        with urllib.request.urlopen(url) as in_stream, open(file_path, "wb") as out_file:
            shutil.copyfileobj(in_stream, out_file)

    dataset = datasets.load_dataset(
        "text",
        data_files={split: file_path},
        split=split,
        cache_dir=cache_dir,
    )
    return dataset.rename_column("text", "sentence")


def get_lm1b_dataset(cache_dir, split):
    if split == "validation":
        split = "test"
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported LM1B split: {split}")

    raw_cache_dir = os.path.join(cache_dir, "lm1b", "raw_data")
    utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)

    archive_path = os.path.join(
        raw_cache_dir, "1-billion-word-language-modeling-benchmark-r13output.tar.gz"
    )
    extracted_root = os.path.join(raw_cache_dir, LM1B_TOP_LEVEL_DIR)
    target_dir = LM1B_TRAIN_DIR if split == "train" else LM1B_HELDOUT_DIR
    target_root = os.path.join(raw_cache_dir, target_dir)

    if not utils.fsspec_exists(target_root):
        if not utils.fsspec_exists(archive_path):
            LOGGER.info("Downloading lm1b from %s", LM1B_URL)
            with urllib.request.urlopen(LM1B_URL) as in_stream, open(
                archive_path, "wb"
            ) as out_file:
                shutil.copyfileobj(in_stream, out_file)

        LOGGER.info("Extracting lm1b %s split from %s", split, archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.name.startswith(target_dir + "/")
            ]
            archive.extractall(raw_cache_dir, members=members)

    pattern = os.path.join(target_root, "news.en-*")
    file_paths = sorted(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError(f"No LM1B files found for split={split} under {target_root}")

    dataset = datasets.load_dataset(
        "text",
        data_files={split: file_paths},
        split=split,
        cache_dir=cache_dir,
    )
    return dataset


def get_text8_dataset(cache_dir, max_seq_length=256, drop_last=True, crop_train=False):
    """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
          (default: 256, as in D3PM codebase.)
      drop_last: bool, whether to drop the last incomplete
          batch. (default: True, as in D3PM codebase.)
      crop_train: bool, whether to subsample contiguous
          subsequences from training example. serves to
          make sure transformer models with absolute position
          embeddings do not have incorrect position-wise
          marginals. (default: False, but necessary to match D3PM AR)

    Returns:
      dataset: dataset.DatasetDict, with keys 'train',
          'valid', 'test'.
    """
    url = "http://mattmahoney.net/dc/text8.zip"
    if not crop_train:
        cache_dir = f"{cache_dir}/text8"
    else:
        cache_dir = f"{cache_dir}/text8-crop-train"
    split_names = ["train", "validation", "test"]
    if not all(
        [utils.fsspec_exists(os.path.join(cache_dir, split)) for split in split_names]
    ):
        # Check if raw data exists
        raw_cache_dir = os.path.join(cache_dir, "raw_data")
        if not all(
            [
                utils.fsspec_exists(os.path.join(raw_cache_dir, f"text8.{split}.txt"))
                for split in split_names
            ]
        ):
            if not utils.fsspec_exists(os.path.join(raw_cache_dir, "text8.zip")):
                utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
                LOGGER.info("Downloading text8 from URL {}.".format(url))
                with (
                    urllib.request.urlopen(url) as in_stream,
                    open(os.path.join(raw_cache_dir, "text8.zip"), "wb") as out_file,
                ):
                    shutil.copyfileobj(in_stream, out_file)

            with fsspec.open(os.path.join(raw_cache_dir, "text8.zip"), "rb") as f:
                rawdata = zipfile.ZipFile(f).read("text8").decode("utf-8")

            # Splits taken from D3PM codebase
            splits = {
                "train": rawdata[:90000000],
                "validation": rawdata[90000000:95000000],
                "test": rawdata[95000000:],
            }

            for split, data in splits.items():
                _path = os.path.join(raw_cache_dir, f"text8.{split}.txt")
                with fsspec.open(_path, "w") as f:
                    f.write(data)
        else:
            splits = {}
            for split in split_names:
                _path = os.path.join(raw_cache_dir, f"text8.{split}.txt")
                with fsspec.open(_path, "r") as f:
                    splits[split] = f.read()

        # Chunk and save as datasets.DatasetDict
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        dataset_dict = {}
        for k, v in splits.items():
            if k == "train" and crop_train == True:
                chunk_size = 2 * max_seq_length
            else:
                chunk_size = max_seq_length
            text = list(chunks(v, chunk_size))
            if drop_last and len(text[-1]) < chunk_size:
                text = text[:-1]
            dataset_dict[k] = datasets.Dataset.from_dict({"text": text})
        dataset = datasets.DatasetDict(dataset_dict)
        dataset.save_to_disk(cache_dir)
    else:
        dataset = datasets.load_from_disk(cache_dir)

    return dataset


def _group_texts(examples, block_size, bos, eos):
    # Concatenate all texts.
    concatenated_examples = list(itertools.chain(*examples["input_ids"]))
    total_length = len(concatenated_examples)
    # TODO(yair): look into not dropping the remainder but rather padding it.
    # We drop the small remainder, and if the total_length < block_size - 2
    # we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of
    # this drop, you can customize this part to your needs.
    new_block_size = block_size - 2  # [BOS] and [EOS] to be added
    total_length = (total_length // new_block_size) * new_block_size
    # Split by chunks of max_len.
    result = {}
    _values = []
    _attn_masks = []
    for i in range(0, total_length, new_block_size):
        _values.append([bos] + concatenated_examples[i : i + new_block_size] + [eos])
        _attn_masks.append(torch.ones(block_size))
    result["input_ids"] = _values
    result["attention_mask"] = _attn_masks
    return result


def get_dataset(
    dataset_name,
    tokenizer,
    wrap,
    mode,
    cache_dir,
    insert_eos=True,
    block_size=1024,
    num_proc=len(os.sched_getaffinity(0)),
    streaming=False,
    revision: Optional[str] = None,
):
    eos_tag = ""
    if not insert_eos:
        eos_tag = "_eosFalse"
    if wrap:
        filename = f"{dataset_name}_{mode}_bs{block_size}_wrapped{eos_tag}.dat"
    else:
        filename = f"{dataset_name}_{mode}_bs{block_size}_unwrapped{eos_tag}.dat"
    _path = os.path.join(cache_dir, filename)

    if utils.fsspec_exists(_path):
        LOGGER.info(f"Loading data from: {_path}")
        return datasets.load_from_disk(_path).with_format("torch")
    LOGGER.info(f"Generating new data at: {_path}")
    LOGGER.info(f"{streaming=}")

    crop_train = dataset_name == "text8-crop"
    if mode == "train" and crop_train:
        # double block size for sub-sampling
        block_size *= 2

    if num_proc is not None:
        num_proc = int(num_proc)
        if num_proc <= 1:
            num_proc = None

    direct_split_dataset = False

    if dataset_name == "wikitext103":
        dataset = datasets.load_dataset(
            "wikitext",
            name="wikitext-103-raw-v1",
            split=mode,
            cache_dir=cache_dir,
            revision=revision,
        )
        direct_split_dataset = True
    elif dataset_name == "wikitext2":
        dataset = datasets.load_dataset(
            "wikitext",
            name="wikitext-2-raw-v1",
            split=mode,
            cache_dir=cache_dir,
            revision=revision,
        )
        direct_split_dataset = True
    elif dataset_name == "ptb":
        if streaming:
            raise ValueError("Streaming is not supported for PTB fallback")
        dataset = get_ptb_dataset(cache_dir=cache_dir, split=mode)
        direct_split_dataset = True
    elif dataset_name == "lambada":
        dataset = get_lambada_test_dataset()
        direct_split_dataset = True
    elif dataset_name == "text8":
        assert wrap
        assert revision is None
        dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
    elif dataset_name == "text8-crop":
        assert revision is None
        dataset = get_text8_dataset(
            cache_dir, max_seq_length=block_size, crop_train=True
        )
    elif dataset_name == "openwebtext-train":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[:-100000]",
            cache_dir=cache_dir,
            revision=revision,
            streaming=False,
            num_proc=num_proc,
        )
    elif dataset_name == "openwebtext-valid":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[-100000:]",
            cache_dir=cache_dir,
            revision=revision,
            streaming=False,
            num_proc=num_proc,
        )
    elif dataset_name == "scientific_papers_arxiv":
        if streaming:
            raise ValueError("Streaming is not supported for scientific_papers fallback")
        dataset = get_scientific_papers_dataset(
            cache_dir=cache_dir,
            subset="arxiv",
            split=mode,
            num_proc=num_proc,
        )
        direct_split_dataset = True
    elif dataset_name == "scientific_papers_pubmed":
        if streaming:
            raise ValueError("Streaming is not supported for scientific_papers fallback")
        dataset = get_scientific_papers_dataset(
            cache_dir=cache_dir,
            subset="pubmed",
            split=mode,
            num_proc=num_proc,
        )
        direct_split_dataset = True
    elif dataset_name == "ag_news":
        dataset = datasets.load_dataset(
            "ag_news",
            split=mode,
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision,
        )
        direct_split_dataset = True
    elif dataset_name == "lm1b":
        if streaming:
            raise ValueError("Streaming is not supported for LM1B fallback")
        dataset = get_lm1b_dataset(cache_dir=cache_dir, split=mode)
        direct_split_dataset = True
    elif dataset_name == "dvruette/lm1b":
        dataset = datasets.load_dataset(
            "dvruette/lm1b",
            split=mode,
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision,
        )
        direct_split_dataset = True
    elif dataset_name == "synthetic":
        assert streaming
        assert wrap  # i.e., no pad tokens
        dataset = generate_synthetic_dataset(
            train_dataset_size=100000,
            validation_dataset_size=1024,
            seq_len=32,
            vocab_size=256,
        )
    else:
        dataset = datasets.load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision,
        )

    if direct_split_dataset or dataset_name in ["openwebtext-train", "openwebtext-valid"]:
        data = dataset
    else:
        data = dataset[mode]
        if dataset_name == "synthetic":
            # already tokenized, no further actions required
            return data

    if dataset_name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif dataset_name == "ptb":
        detokenizer = ptb_detokenizer
    elif dataset_name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif dataset_name == "lambada":
        detokenizer = lambada_detokenizer
    elif dataset_name.startswith("scientific_papers"):
        detokenizer = scientific_papers_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text

        return detok

    EOS = tokenizer.encode(tokenizer.eos_token)[0]
    BOS = tokenizer.encode(tokenizer.bos_token)[0]

    def preprocess_and_tokenize(example):
        if dataset_name == "ptb":
            text = example["sentence"]
        elif "scientific_papers" in dataset_name:
            text = example["article"]
        else:
            text = example["text"]

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"

        if wrap:
            tokens = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            if insert_eos:
                tokens = {"input_ids": [t + [EOS] for t in tokens["input_ids"]]}
            # Still missing BOS, but will be added in group_texts
        else:
            tokens = tokenizer(
                text,
                max_length=block_size,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
        return tokens

    if streaming:
        tokenized_dataset = data.map(preprocess_and_tokenize, batched=True)
    else:
        tokenized_dataset = data.map(
            preprocess_and_tokenize,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc="Tokenizing",
        )
    if dataset_name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns("sentence")
    elif "scientific_papers" in dataset_name:
        tokenized_dataset = tokenized_dataset.remove_columns(
            ["article", "abstract", "section_names"]
        )
    elif dataset_name == "ag_news":
        tokenized_dataset = tokenized_dataset.remove_columns(["text", "label"])
    else:
        tokenized_dataset = tokenized_dataset.remove_columns("text")

    if not wrap:
        if not streaming:
            tokenized_dataset.save_to_disk(_path)
        return tokenized_dataset.with_format("torch")

    group_texts = functools.partial(
        _group_texts, block_size=block_size, bos=BOS, eos=EOS
    )
    if streaming:
        chunked_dataset = tokenized_dataset.map(group_texts, batched=True)
    else:
        chunked_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc="Grouping",
        )
        chunked_dataset.save_to_disk(_path)
    chunked_dataset = chunked_dataset.with_format("torch")
    return chunked_dataset


def get_tokenizer(config):
    if config.data.tokenizer_name_or_path == "text8":
        tokenizer = Text8Tokenizer()
    elif config.data.tokenizer_name_or_path == "bert-base-uncased":
        tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=True
        )
    elif config.data.tokenizer_name_or_path == "synthetic":
        tokenizer = SyntheticTokenizer(vocab_size=256)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.data.tokenizer_name_or_path
        )

    if isinstance(tokenizer, transformers.GPT2TokenizerFast) or isinstance(
        tokenizer, transformers.GPT2Tokenizer
    ):
        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id),
        )

    # For wrapped batches:
    #  [BOS] sent1 [EOS] sent2-fragment [EOS]
    #  [BOS] sent2-fragment [EOS] sent3 [EOS]
    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(
                "Tokenizer must have a bos_token or " f"cls_token: {tokenizer}"
            )
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(
                "Tokenizer must have a eos_token " f"or sep_token: {tokenizer}"
            )
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def get_dataloaders(
    config, tokenizer, skip_train=False, skip_valid=False, valid_seed=None
):
    train_num_workers = config.loader.get(
        "train_num_workers", config.loader.num_workers
    )
    eval_num_workers = config.loader.get("eval_num_workers", config.loader.num_workers)
    num_gpus = torch.cuda.device_count()
    assert config.loader.global_batch_size == (
        config.loader.batch_size
        * config.trainer.num_nodes
        * num_gpus
        * config.trainer.accumulate_grad_batches
    )
    if (
        config.loader.global_batch_size
        % (num_gpus * config.trainer.accumulate_grad_batches)
        != 0
    ):
        raise ValueError(
            f"Train Batch Size {config.training.batch_size}"
            f"not divisible by {num_gpus} gpus with accumulation "
            f"{config.trainer.accumulate_grad_batches}."
        )
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
            f"Eval Batch Size for {config.eval.batch_size} "
            f"not divisible by {num_gpus}."
        )
    if skip_train:
        train_set = None
    else:
        train_set = get_dataset(
            config.data.train,
            tokenizer,
            mode="train",
            wrap=config.data.wrap,
            insert_eos=config.data.insert_train_eos,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            streaming=config.data.streaming,
            num_proc=train_num_workers,
            revision=config.data.get("train_revision", None),
        )

    if config.data.valid in ["text8", "dvruette/lm1b", "lm1b", "ag_news"]:
        validation_split = "test"
    else:
        validation_split = "validation"
    if skip_valid:
        valid_set = None
    else:
        valid_set = get_dataset(
            config.data.valid,
            tokenizer,
            wrap=config.data.wrap,
            mode=validation_split,
            cache_dir=config.data.cache_dir,
            insert_eos=config.data.insert_valid_eos,
            block_size=config.model.length,
            streaming=config.data.streaming,
            num_proc=eval_num_workers,
            revision=config.data.get("valid_revision", None),
        )

    if skip_train:
        train_loader = None
    else:
        persistent_workers = train_num_workers > 0
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.loader.batch_size,
            num_workers=train_num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=not config.data.streaming,
            persistent_workers=persistent_workers,
        )
        train_loader.tokenizer = tokenizer
    if skip_valid:
        valid_loader = None
    else:
        if valid_seed is None:
            shuffle_valid = False
            generator = None
        else:
            shuffle_valid = True
            generator = torch.Generator().manual_seed(valid_seed)
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.loader.eval_batch_size,
            num_workers=eval_num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=shuffle_valid,
            generator=generator,
        )
        # Will be used in generative perplexity calculation
        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

    def __init__(self, *args, generator=None, **kwargs):
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called beforehand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        kwargs.pop("shuffle", None)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"random_state": self.generator.get_state(), "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get("random_state"))
        self.counter = state_dict["counter"]
        # self.start_counter = self.counter
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.

    def __iter__(self) -> typing.Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter :]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"epoch": self.epoch, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.counter = state_dict["counter"]
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter :]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0
