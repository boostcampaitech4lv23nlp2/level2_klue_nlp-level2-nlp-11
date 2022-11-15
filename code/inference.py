import os
import pickle as pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from args import get_args
from klue.dataloader import RE_Dataset, load_data, tokenized_dataset
from klue.utils import num_to_label, set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(
        tokenized_sent, batch_size=conf.train.batch_size, shuffle=False
    )
    model.eval()

    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                token_type_ids=data["token_type_ids"].to(device),
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def main(args, conf):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    set_seed(conf.utils.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(conf.model.model_name)

    ## load my model
    # TODO: model name 이 정확하게 명시된 dir 필요
    model = AutoModelForSequenceClassification.from_pretrained(conf.path.load_model)
    model.parameters
    model.to(device)

    # TODO: load dataset 통합
    test_dataset = load_data(conf.path.test_path)
    test_label = list(map(int, test_dataset["label"].values))

    test_id = test_dataset["id"]

    # test_id, test_dataset, test_label = tokenized_dataset(test_dataset, tokenizer)
    # -- tokenized_dataset is return token_dict(input_ids, token_type_ids, attention_mask) --

    test_dataset = tokenized_dataset(test_dataset, tokenizer)
    test_dataset = RE_Dataset(test_dataset, test_label)

    ## predict answer
    pred_answer, output_prob = inference(
        model, test_dataset, device
    )  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    ## make csv file with predicted answer
    #########################################################

    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )

    output.to_csv(
        "../dataset/prediction/submission.csv",
        index=False
        # "./prediction/submission.csv", index=False
    )  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print("---- Finish! ----")


if __name__ == "__main__":
    args, conf = get_args(mode="test")
    # print(args, conf)
    main(args, conf)
