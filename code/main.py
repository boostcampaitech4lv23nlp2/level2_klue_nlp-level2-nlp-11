import argparse

import inference
import torch
import train
from klue.utils import set_conf, set_seed
from omegaconf import OmegaConf

if __name__ == "__main__":
    """
    argparse 파라미터 설명
        --mode -m : 실행 모드를 지정
                    train : [train, t] 모델 학습
                    inference : [inference, i ] 모델 예측
        --config -c : config.yaml 의 file_name
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", required=True)
    parser.add_argument("--config", "-c", type=str, default="base_config")
    args, sweep_args = parser.parse_known_args()

    # pip install omegaconf 부터
    conf = OmegaConf.load(f"../config/{args.config}.yaml")
    conf.merge_with_dotlist(sweep_args)

    set_seed(conf.utils.seed)
    set_conf(conf)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == "train" or args.mode == "t":
        train.train(conf, device)

    elif args.mode == "inference" or args.mode == "i":
        inference.main(conf, device)
    else:
        raise Exception(
            "모드를 다시 설정해주세요.\ntrain mode: t,\ttrain\ninference mode: i,\tinference"
        )
