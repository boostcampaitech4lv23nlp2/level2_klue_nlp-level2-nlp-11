import sys
import argparse


def get_args(mode="train"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', default=42, type=int)
    # 유지관리
    parser.add_argument('--version', default='', type=str)
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--save_model', default=True)
    parser.add_argument('--wandb', default=True)
    parser.add_argument('--eval_strategy', type=str, default='epoch')
    # Hyperparameter
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    # Path
    parser.add_argument('--train_path', default='../dataset/train/train.csv')
    parser.add_argument('--dev_path', default='../dataset/test/test_data.csv')
    parser.add_argument('--test_path', default='../dataset/test/test_data.csv')
    parser.add_argument('--predict_path', default='../dataset/results/submission.csv')
    parser.add_argument('--data_dir', default= '../data')
    parser.add_argument('--model_dir', default= '../dataset/best_model')
    parser.add_argument('--save_dir', default= '../dataset/results')
    parser.add_argument('--logs_dir', default='../dataset/logs')
    
    return parser.parse_args(sys.argv[1:])