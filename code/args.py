import argparse
import sys
from typing import Tuple

from omegaconf import OmegaConf


def get_args(mode="train") -> Tuple[argparse.Namespace, OmegaConf.Dictconfig]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="base_config")
    args, _ = parser.parse_known_args()

    # pip install omegaconf 부터
    conf = OmegaConf.load(f"../config/{args.config}.yaml")

    return args, conf
