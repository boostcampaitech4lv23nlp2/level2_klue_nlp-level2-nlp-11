import pickle as pickle
from pathlib import Path

import torch
import wandb
from klue.dataloader import get_dataset , set_tokenizer
from klue.metric import compute_metrics, klue_re_auprc, klue_re_micro_f1
from klue.trainer import FocallossTrainer
from klue.utils import set_MODEL_NAME, set_seed
import Model
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, BertTokenizer, EarlyStoppingCallback,
                          RobertaConfig, RobertaForSequenceClassification,
                          RobertaTokenizer, Trainer, TrainingArguments)


def train(conf, device) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model and tokenizer
    # TODO: BETTER WAY TO SET DIRECTORIES!!!!
    DISPLAY_NAME = f"lr-{conf.train.learning_rate:5f}_{conf.wandb.annotation}"
    MODEL_NAME = set_MODEL_NAME(conf.model.model_name, conf.path.save_dir)
    SAVE_DIR = Path(conf.path.save_dir) / MODEL_NAME
    LOG_DIR = Path(conf.path.logs_dir) / MODEL_NAME
    MODEL_DIR = Path(conf.path.model_dir) / MODEL_NAME
    WANDB_DIR = Path(conf.path.wandb_dir)
    # Initialize wandb
    wandb.init(
        project=f"{conf.wandb.exp_name}",
        entity="we-fusion-klue",
        dir=WANDB_DIR,
        name=f"{DISPLAY_NAME}",
        group=f"{conf.model.model_name.replace('/','_')}",
    )
    
    #set_tokenizerr
    tokenizer = AutoTokenizer.from_pretrained(conf.model.model_name)
    add_token = [] #['<s>', '</s>' , '<person>' , '<location>']

    tokenizer, new_vocab_size = set_tokenizer(tokenizer, add_token)
    print(add_token)
    print(new_vocab_size)

    # load dataset
    train_dataset = get_dataset(conf.path.train_path, tokenizer)
    valid_dataset = get_dataset(conf.path.valid_path, tokenizer)

    assert conf.model.model_type in ['BaseModel' ,'CustomModel'],  "model.model_type  is not ['BaseModel' , 'CustomModel']!.  please check config.yaml"

    model = Model.load_model(conf.model.model_type ,conf.model.model_name ,new_vocab_size)
    
    model = model.get_model()

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
        # early stopping
        metric_for_best_model="eval_micro f1 score",
    )
    
    assert conf.model.trainer in ['Base', 'Focalloss'] , "model.trainer  is not ['Base' , 'Focalloss'] please check config.yaml"

    if conf.model.trainer == 'Base' :
        print('base_trainer')
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,             # evaluation dataset
            compute_metrics=compute_metrics         # define metrics function
        )

    elif conf.model.trainer == 'Focalloss' :
        print('Focalloss_trainer')
        trainer = FocallossTrainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=valid_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # define metrics function
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # set patience
            gamma=conf.focalloss.gamma,  # set focalloss gamma.
    )

    # train model
    trainer.train()
    model.save_pretrained(MODEL_DIR)


def k_train() -> None:
    """
    구현예정
    """
