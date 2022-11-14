import pickle as pickle

import torch
from args import get_args
from klue.dataloader import (RE_Dataset, load_data, preprocessing_dataset,
                             tokenized_dataset)
from klue.metric import compute_metrics, klue_re_auprc, klue_re_micro_f1
from klue.utils import label_to_num, set_seed
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, BertTokenizer, RobertaConfig,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          Trainer, TrainingArguments)


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model and tokenizer
    # TODO: BETTER WAY TO SET DIRECTORIES!!!!
    MODEL_NAME = f"{args.model_name.replace('/','_')}_{args.version}"
    SAVE_DIR = f"{args.save_dir}/{MODEL_NAME}"
    LOG_DIR = f"{args.logs_dir}/{MODEL_NAME}"
    MODEL_DIR = f"{args.model_dir}/{MODEL_NAME}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # load dataset
    train_dataset = load_data(args.train_path)
    # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset["label"].values)
    # dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    train_dataset = RE_Dataset(tokenized_train, train_label)
    # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=model_config
    )
    print(model.config)
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=SAVE_DIR,  # output directory
        save_total_limit=5,  # number of total save model.
        save_steps=500,  # model saving step.
        num_train_epochs=args.max_epoch,  # total number of training epochs
        learning_rate=args.learning_rate,  # learning_rate
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=LOG_DIR,  # directory for storing logs
        logging_steps=100,  # log saving step.
        evaluation_strategy=args.eval_strategy,  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=500,  # evaluation step.
        seed=args.seed,
        fp16=True
        # TODO: evaluation_strategy 에 맞춰 변경이 필요함.
        # load_best_model_at_end = True
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=train_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained(MODEL_DIR)


def main():
    set_seed(args.seed)
    train()


if __name__ == "__main__":
    args = get_args(mode="train")
    main()
