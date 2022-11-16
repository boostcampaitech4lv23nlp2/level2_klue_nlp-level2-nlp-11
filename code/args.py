import argparse
import sys
from typing import Tuple

from omegaconf import OmegaConf, dictconfig


def get_args(
    mode="train",
) -> Tuple[argparse.Namespace, dictconfig.DictConfig]:
    # TODO : 함수 독스트링을 작성해주세요.
    # TODO : mode parameter의 용도가 불분명합니다.
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="base_config")
    args, _ = parser.parse_known_args()

    # pip install omegaconf 부터
    conf = OmegaConf.load(f"../config/{args.config}.yaml")

    return args, conf
