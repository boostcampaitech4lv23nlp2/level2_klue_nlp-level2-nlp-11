import pickle as pickle

import torch
import wandb
from klue.dataloader import get_dataset
from klue.metric import compute_metrics, klue_re_auprc, klue_re_micro_f1
from klue.utils import FocalLoss, label_to_num, set_seed
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, BertTokenizer, RobertaConfig,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          Trainer, TrainingArguments)


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
    # TODO : trainer와 관련된 클래스 또는 함수를 "code/klue/trainer.py"로 옮겨주세요!
    class FocallossTrainer(Trainer):
        # gamma, alpha를 직접 설정할 수 있도록 코드를 개선하였습니다.
        # 다만 alpha는 int값을 넣을시 gather와 관련하여 오류가 발생힙니다.
        def __init__(self, gamma: int = 5, alpha: int = None, **kwargs):
            super().__init__(**kwargs)
            self.gamma = gamma
            self.alpha = alpha  # alpha는 안쓰는것을 추천한다.쓰는 순간 오류가 발생하는 이슈가 있음.

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # loss_fct = MSELoss()
            # loss = loss_fct(logits.squeeze(), labels.squeeze())
            loss = FocalLoss(gamma=self.gamma, alpha=self.alpha)(
                logits.squeeze(), labels.squeeze()
            )
            return (loss, outputs) if return_outputs else loss

    trainer = FocallossTrainer(
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
