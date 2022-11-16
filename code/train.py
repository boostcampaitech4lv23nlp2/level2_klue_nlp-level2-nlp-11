import pickle as pickle

import wandb
import torch
from klue.dataloader import get_dataset
from klue.metric import compute_metrics, klue_re_auprc, klue_re_micro_f1
from klue.utils import label_to_num, set_seed
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)


def train(conf) -> None:
    wandb.init(project="test-project", entity="we-fusion-klue")
    set_seed(conf.utils.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model and tokenizer
    # TODO: BETTER WAY TO SET DIRECTORIES!!!!
    MODEL_NAME = f"{conf.model.model_name.replace('/','_')}_{conf.maintenance.version}"
    SAVE_DIR = f"{conf.path.save_dir}/{MODEL_NAME}"
    LOG_DIR = f"{conf.path.logs_dir}/{MODEL_NAME}"
    MODEL_DIR = f"{conf.path.model_dir}/{MODEL_NAME}"

    tokenizer = AutoTokenizer.from_pretrained(conf.model.model_name)

    # load dataset
    train_dataset = get_dataset(conf.path.train_path, tokenizer)
    valid_dataset = get_dataset(conf.path.valid_path, tokenizer)
    print(device)

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(conf.model.model_name)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(
        conf.model.model_name, config=model_config
    )

    print(model.config)
    model.parameters
    model.to(device)
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        # set seed
        seed=conf.utils.seed,
        # dir setting
        output_dir=SAVE_DIR,  # output directory
        logging_dir=LOG_DIR,  # directory for storing logs
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
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained(MODEL_DIR)


def k_train() -> None:
    """
    구현예정
    """
