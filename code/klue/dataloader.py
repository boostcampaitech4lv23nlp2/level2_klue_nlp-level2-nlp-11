import os
import pathlib
import pickle as pickle

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from . import utils


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset: pd.DataFrame, labels: np.ndarray):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = {
            key: val[idx].clone().detach() for key, val in self.pair_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def preprocessing_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = i[1:-1].split(",")[0].split(":")[1]
        j = j[1:-1].split(",")[0].split(":")[1]

        subject_entity.append(i)
        object_entity.append(j)
    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )
    return out_dataset


def load_data(dataset_dir: pathlib.Path) -> pd.DataFrame:
    """csv 파일을 경로에 맞게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def tokenized_dataset(dataset: pd.DataFrame, tokenizer: AutoTokenizer) -> torch.Tensor:
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
    return tokenized_sentences


def get_dataset(data_path: pathlib.Path, tokenizer: AutoTokenizer) -> RE_Dataset:
    """데이터셋을 Trainer에 넣을 수 있도록 처리하여 리턴합니다.

    Args:
        data_path (str): 가져올 데이터의 주소입니다.
        tokenizer (AutoTokenizer): 데이터를 토큰화할 토크나이저입니다.

    Returns:
        pd.DataFrame: _description_
    """
    dataset = load_data(data_path)
    dataset_label = utils.label_to_num(dataset["label"].values)
    # tokenizing dataset
    dataset_tokens = tokenized_dataset(dataset, tokenizer)
    # make dataset for pytorch.
    dataset = RE_Dataset(dataset_tokens, dataset_label)
    return dataset


def get_test_dataset(data_path: pathlib.Path, tokenizer: AutoTokenizer) -> RE_Dataset:
    """데이터셋을 Trainer에 넣을 수 있도록 처리하여 리턴합니다.

    Args:
        data_path (str): 가져올 데이터의 주소입니다.
        tokenizer (AutoTokenizer): 데이터를 토큰화할 토크나이저입니다.

    Returns:
        pd.DataFrame: _description_
    """
    dataset = load_data(data_path)
    dataset_label = list(map(int, dataset["label"].values))
    dataset_id = dataset["id"]
    # tokenizing dataset
    dataset_tokens = tokenized_dataset(dataset, tokenizer)
    # make dataset for pytorch.
    dataset = RE_Dataset(dataset_tokens, dataset_label)
    return dataset_id, dataset, dataset_label
