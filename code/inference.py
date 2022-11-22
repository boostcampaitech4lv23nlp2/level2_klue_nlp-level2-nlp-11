import os
import pickle as pickle

import Model
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from klue.dataloader import set_tokenizer, load_dataloader

from klue.utils import num_to_label, set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


# TODO : ADD TYPE HINT!
def inference(model, tokenized_sent, batch_size, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size, shuffle=False)
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


def main(conf, device):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    MODEL_NAME = f"{conf.model.model_name.replace('/','_')}_{conf.maintenance.version}"
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(conf.model.model_name)
    tokenizer, new_vocab_size = set_tokenizer(tokenizer)
    print(new_vocab_size)

    ## load my model
    # TODO: model name 이 정확하게 명시된 dir 필요
    
    # TODO: load dataset 통합
    assert conf.data.data_loader in ['BaseDataLoader' ,'CustomDataLoader'], "data.data_loader is not ['BaseDataLoader' , 'CustomDataLoader']!.  please check config.yaml"

    # load dataset
    test_id, test_dataset, test_label = load_dataloader(conf.data.data_loader, conf.path.test_path, tokenizer).get_test_dataset()

    # Model load
    assert conf.model.model_type in ['BaseModel' ,'CustomModel'],  "model.model_type  is not ['BaseModel' , 'CustomModel']!.  please check config.yaml"

    model = Model.load_model(conf.model.model_type, conf.path.load_model , new_vocab_size)
    model = model.get_model()
    print(model)
    print(model.config)
    model.parameters
    model.to(device)

    ## predict answer
    pred_answer, output_prob = inference(
        model, test_dataset, conf.train.per_device_train_batch_size, device
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
    # if not os.path.isdir(f"{conf.path.predict_dir}/{conf.model.model_name.replace('/','_')}") :
    #     os.mkdir(f"{conf.path.predict_dir}/{conf.model.model_name.replace('/','_')}")

    output.to_csv(
        f"{conf.path.predict_dir}/{MODEL_NAME}.csv",
        index=False
        # "./prediction/submission.csv", index=False
    )  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print(f"saved path : {conf.path.predict_dir}/{MODEL_NAME}.csv")
    print("---- Finish! ----")
