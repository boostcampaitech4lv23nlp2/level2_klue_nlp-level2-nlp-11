import pickle as pickle
import statistics
from pathlib import Path

import klue.dataloader
import klue.Model
import pandas as pd
import torch
import wandb
from inference import inference
from klue.dataloader import (load_dataloader, load_dataloader_kfold,
                             set_tokenizer)
from klue.metric import compute_metrics
from klue.Model import load_model
from klue.trainer import FocallossTrainer
from klue.utils import get_DIR, num_to_label, set_MODEL_NAME
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoTokenizer, EarlyStoppingCallback, Trainer,
                          TrainingArguments)


def train_step(conf, device, new_vocab_size, train_dataset, valid_dataset) -> None:

    assert hasattr(
        klue.Model, conf.model.model_type
    ), f"{conf.model.model_type} is not in klue/Model.py"

    model = load_model(conf.model.model_type, conf.model.model_name, new_vocab_size)
    model = model.get_model()

    print(model)
    print(model.config)
    model.parameters

    model.to(device)
    # TODO : training_args argument abstraction
    # training_Args의 대부분은 고정되어 변하지 않으므로
    # 이를 따로 파일을 만들어서 관리하거나 하면 좋을 것 같습니다.
    training_args = TrainingArguments(
        # set seed
        seed=conf.utils.seed,
        # dir setting
        output_dir=get_DIR("SAVE_DIR"),  # output directory
        logging_dir=get_DIR("LOG_DIR"),  # directory for storing logs
        save_total_limit=5,  # number of total save model.
        # strategy
        evaluation_strategy=conf.maintenance.eval_strategy,  # evaluation strategy to adopt during training
        save_strategy=conf.maintenance.eval_strategy,
        # set steps(if strategy is "epoch" or "no", save_steps and eval_steps not work.)
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        save_steps=500,  # model saving step.
        eval_steps=500,  # evaluation step.
        logging_steps=100,  # log saving step.
        # optim parameter.(default optim is adamw_hf)
        weight_decay=0.01,  # strength of weight decay
        # etc..
        fp16=True,
        load_best_model_at_end=True,
        # wandb
        report_to="wandb",
        # train
        **conf.train,  # use dict unpacking.
        # early stopping
        metric_for_best_model="eval_micro f1 score",
    )

    print(
        conf.data.data_loader, new_vocab_size, conf.model.model_type, conf.model.trainer
    )
    # TODO: Improve feature that load trainers.
    if conf.model.trainer == "Base":
        print("base_trainer")
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=valid_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # define metrics function
        )

    elif conf.model.trainer == "Focalloss":
        print("Focalloss_trainer")
        trainer = FocallossTrainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=valid_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # define metrics function
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=5)
            ],  # set patience
            gamma=conf.focalloss.gamma,  # set focalloss gamma.
        )

    # train model
    trainer.train()
    model.save_pretrained(get_DIR("MODEL_DIR"))
    # saving wandb files
    OmegaConf.save(config=conf, f=Path(wandb.run.dir) / Path("train_config.yaml"))
    print(f"save model path : {get_DIR('MODEL_DIR')}")

    return trainer


def train(conf, device) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(
        project=f"{conf.wandb.exp_name}",
        entity="we-fusion-klue",
        dir=get_DIR("WANDB_DIR"),
        name=f"{get_DIR('DISPLAY_NAME')}",
        group=f"{conf.model.model_name.replace('/','_')}",
    )

    # set_tokenizerr
    tokenizer = AutoTokenizer.from_pretrained(conf.model.model_name)
    tokenizer, new_vocab_size = set_tokenizer(tokenizer)

    assert hasattr(
        klue.dataloader, conf.data.data_loader
    ), f"{conf.data.data_loader} is not in klue/dataloader.py"

    if conf.k_fold.use_k_fold:
        # get full dataset
        train_all_df = pd.read_csv("../dataset/train/train.csv")
        st_kfold = StratifiedKFold(n_splits=conf.k_fold.num_split)

        valid_losses = []
        valid_f1_scores = []
        valid_auprcs = []

        fold_cnt = 1

        for train_idx, valid_idx in st_kfold.split(train_all_df, train_all_df["label"]):

            print(f"!!!!!!!! start {fold_cnt} fold !!!!!!!")

            train_df = train_all_df.iloc[
                train_idx,
            ]
            valid_df = train_all_df.iloc[
                valid_idx,
            ]

            # load dataset
            train_dataset = load_dataloader_kfold(
                None, train_df, tokenizer
            ).get_dataset()
            valid_dataset = load_dataloader_kfold(
                None, valid_df, tokenizer
            ).get_dataset()

            print("train_dataset :", len(train_dataset))
            print("valid_dataset :", len(valid_dataset))

            trainer = train_step(
                conf, device, new_vocab_size, train_dataset, valid_dataset
            )

            print("-----------------")
            print("eval_loss: ", trainer.state.log_history[-2]["eval_loss"])
            print(
                "eval_micro f1 score: ",
                trainer.state.log_history[-2]["eval_micro f1 score"],
            )
            print("eval_auprc: ", trainer.state.log_history[-2]["eval_auprc"])
            print("-----------------")
            valid_losses.append(trainer.state.log_history[-2]["eval_loss"])
            valid_f1_scores.append(trainer.state.log_history[-2]["eval_micro f1 score"])
            valid_auprcs.append(trainer.state.log_history[-2]["eval_auprc"])

            print("----------inference -------------")

            # make inference kfold csv output
            test_csv = pd.read_csv(conf.path.test_path)

            test_id, test_dataset, test_label = load_dataloader_kfold(
                None, test_csv, tokenizer
            ).get_test_dataset()

            model = load_model(
                conf.model.model_type, get_DIR("MODEL_DIR"), new_vocab_size
            )
            model = model.get_model()
            model.parameters
            model.to(device)

            ## predict answer
            pred_answer, output_prob = inference(
                model, test_dataset, conf.train.per_device_train_batch_size, device
            )  # model에서 class 추론
            pred_answer = num_to_label(pred_answer)

            output = pd.DataFrame(
                {
                    "id": test_id,
                    "pred_label": pred_answer,
                    "probs": output_prob,
                }
            )

            MODEL_NAME = f"{conf.model.model_name.replace('/','_')}_{conf.maintenance.version}_{fold_cnt}"
            output.to_csv(
                f"{conf.path.predict_dir}/{MODEL_NAME}.csv", index=False
            )  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
            #### 필수!! ##############################################
            print(f"saved path : {conf.path.predict_dir}/{MODEL_NAME}.csv")
            fold_cnt += 1

        print("last eval_loss mean:", statistics.mean(valid_losses))
        print("last eval_micro f1 score mean:", statistics.mean(valid_f1_scores))
        print("last eval_auprc mean:", statistics.mean(valid_auprcs))
        print("last eval_loss :", valid_losses)
        print("last eval_micro f1 score :", valid_f1_scores)
        print("last eval_auprc:", valid_auprcs)

    else:
        # load dataset
        train_dataset = load_dataloader(
            conf.data.data_loader, conf.path.train_path, tokenizer
        ).get_dataset()
        valid_dataset = load_dataloader(
            conf.data.data_loader, conf.path.valid_path, tokenizer
        ).get_dataset()

        print("train_dataset :", len(train_dataset))
        print("valid_dataset :", len(valid_dataset))

        train_step(
            conf=conf,
            device=device,
            new_vocab_size=new_vocab_size,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        )
