import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

class BaseModel():
    def __init__(
        self, model_name , new_vocab_size
    ):  # 새로운 vocab 사이즈 설정
        self.model_name = model_name
        
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.num_labels = 30

        self.plm = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, config =model_config
        )

        self.plm.resize_token_embeddings(new_vocab_size)

    def get_model(self) :
        return self.plm

class CustomModel():
    def __init__(
        self, model_name , new_vocab_size
    ):  # 새로운 vocab 사이즈 설정
    
        self.model_name = model_name
        
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.num_labels = 30

        self.plm = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, config =model_config
        )

        self.plm.resize_token_embeddings(new_vocab_size)

    def get_model(self) :
        return self.plm

def load_model(model_type : str , model_name : str, new_vocab_size : int) :
    model_config = {
    "BaseModel": BaseModel(model_name,new_vocab_size),
    "CustomModel": CustomModel(model_name,new_vocab_size)
    }
    return model_config[model_type]