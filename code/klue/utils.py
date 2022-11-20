import math
import os
import pickle as pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


def set_seed(random_seed: int) -> None:
    print(f"Set global seed {random_seed}")
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


def label_to_num(label: np.ndarray) -> list:
    num_label = []
    with open("/opt/ml/code/dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def num_to_label(label: np.ndarray) -> list:
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open("/opt/ml/code/dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def set_MODEL_NAME(model_name: str, save_dir_path: str) -> Path:
    """동일한 모델을 사용할 때, 버전을 일일이 입력해주어야 하는 문제를 해결하기 위한 함수입니다.
    만약 save_dir에 동일한 모델을 저장한 경우가 있을경우, 마지막 version에 1을 더한 path를 추가합니다.


    Args:
        model_name (str): 모델의 이름
        save_dir_path (str): 저장 모델의 경우.

    Returns:
        Path: _description_
    """
    # pre-processing
    model_name = model_name.replace("/", "_")
    # version check
    version = 1
    MODEL_NAME = Path(save_dir_path) / Path(model_name) / str(version)
    while MODEL_NAME.exists():
        version += 1
        MODEL_NAME = Path(save_dir_path) / Path(model_name) / str(version)
    return Path(model_name) / str(version)



if __name__ == "__main__":
    print(set_MODEL_NAME("klue/roberta-small", "../dataset/results"))
    