from collections import OrderedDict

from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class BaseModel:

    """Base Model 입니다

    Args:
        model_name (str, dir_path): pretrained 모델 이름 또는 load_model dir
        new_vocab_size (_type_): 새로운 vocab size
    func :
        get_model : 모델을 반환 합니다!
    """

    def __init__(self, model_name, new_vocab_size):  # 새로운 vocab 사이즈 설정

        self.model_name = model_name

        model_config = AutoConfig.from_pretrained(model_name)
        model_config.num_labels = 30

        self.plm = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, config=model_config
        )

        self.plm.resize_token_embeddings(new_vocab_size)

    def get_model(self):
        return self.plm


class CustomModel:

    """CustomModel 입니다

    Args:
        model_name (str, dir_path): pretrained 모델 이름 또는 load_model dir
        new_vocab_size (_type_): 새로운 vocab size
    func :
        get_model : 모델을 반환 합니다!
    """

    def __init__(self, model_name, new_vocab_size):  # 새로운 vocab 사이즈 설정

        self.model_name = model_name

        model_config = AutoConfig.from_pretrained(model_name)
        model_config.num_labels = 30

        self.plm = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, config=model_config
        )

        self.custom_headclassfier = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(30, 1024)),
                    ("dropout", nn.Dropout(0.1)),
                    ("out_proj", nn.Linear(1024, 30)),
                ]
            )
        )

        self.plm.add_module("custom_headclassfier", self.custom_headclassfier)

        self.plm.resize_token_embeddings(new_vocab_size)

    def get_model(self):
        return self.plm


def load_model(model_type: str, model_name: str, new_vocab_size: int):
    """ model_type 해당되는 클래스 모델을 반환합니다

    Args:
        model_type (str): 가져올 모델 클래스
        model_name (str, dir_path): pretrained 모델 이름 또는 load_model dir
        new_vocab_size (int): 새로운 vocab size

    Returns:
        model_config: 해당되는 모델 클래스를 반환합니다
    """
    model_config = {
        "BaseModel": BaseModel(model_name, new_vocab_size),
        "CustomModel": CustomModel(model_name, new_vocab_size),
    }
    return model_config[model_type]
