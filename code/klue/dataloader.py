import os
import pathlib
import pickle as pickle
from functools import partial

import numpy as np
import pandas as pd
import torch
from klue.sure import change_type, verbalize_label
from transformers import AutoTokenizer

from . import utils


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset: pd.DataFrame, labels: np.ndarray):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = {
            key: val[idx].clone().detach() for key, val in self.pair_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


class BaseDataLoader:
    """BaseLine DataLoader 입니다.

    Args:
        load_data :  (str): 가져올 데이터의 주소입니다.
        tokenizer (AutoTokenizer): 데이터를 토큰화할 토크나이저입니다.

    func :
        1)preprocessing_dataset(pd.DataFrame) : 데이터 전처리를 합니다 return pd.DataFrame
        2)tokenized_dataset(pd.DataFrame, tokenizer: AutoTokenizer) : 전처리된 데이터셋을 tokenized_dataset으로 반환 합니다
        3)get_dataset : train, valid 데이터셋을 토큰화 셋으로 변환한 뒤 RE_Dataset 형태로 반환 합니다
        4)get_test_dataset :  test 데이터셋을 토큰화 셋으로 변환한 뒤 RE_Dataset 형태로 반환 합니다
    """

    def __init__(self, data_path: pathlib.Path, tokenizer: AutoTokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer

    def preprocessing_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        subject_entity = []
        object_entity = []
        for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
            i = i[1:-1].split(",")[0].split(":")[1]
            j = j[1:-1].split(",")[0].split(":")[1]

            subject_entity.append(i)
            object_entity.append(j)
        out_dataset = pd.DataFrame(
            {
                "id": dataset["id"],
                "sentence": dataset["sentence"],
                "subject_entity": subject_entity,
                "object_entity": object_entity,
                "label": dataset["label"],
            }
        )

        return out_dataset

    def tokenized_dataset(
        self, dataset: pd.DataFrame, tokenizer: AutoTokenizer
    ) -> torch.Tensor:
        """tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = []
        for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
            temp = ""
            temp = e01 + "[SEP]" + e02
            concat_entity.append(temp)

        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset["sentence"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
        return tokenized_sentences

    def get_dataset(self) -> RE_Dataset:
        """데이터셋을 Trainer에 넣을 수 있도록 처리하여 리턴합니다.

        Args:
            data_path (str): 가져올 데이터의 주소입니다.
            tokenizer (AutoTokenizer): 데이터를 토큰화할 토크나이저입니다.

        Returns:
            pd.DataFrame: _description_
        """
        pd_dataset = pd.read_csv(self.data_path)
        dataset = self.preprocessing_dataset(pd_dataset)

        dataset_label = utils.label_to_num(dataset["label"].values)

        # tokenizing dataset
        dataset_tokens = self.tokenized_dataset(dataset, self.tokenizer)
        # make dataset for pytorch.
        dataset = RE_Dataset(dataset_tokens, dataset_label)
        return dataset

    def get_test_dataset(self) -> RE_Dataset:
        """데이터셋을 Trainer에 넣을 수 있도록 처리하여 리턴합니다.

        Args:
            data_path (str): 가져올 데이터의 주소입니다.
            tokenizer (AutoTokenizer): 데이터를 토큰화할 토크나이저입니다.

        Returns:
            pd.DataFrame: _description_
        """
        pd_dataset = pd.read_csv(self.data_path)
        dataset = self.preprocessing_dataset(pd_dataset)

        dataset_label = list(map(int, dataset["label"].values))
        dataset_id = dataset["id"]

        # tokenizing dataset
        dataset_tokens = self.tokenized_dataset(dataset, self.tokenizer)
        # make dataset for pytorch.
        dataset = RE_Dataset(dataset_tokens, dataset_label)
        return dataset_id, dataset, dataset_label


class CustomDataLoader(BaseDataLoader):
    """Custom DataLoader 입니다.

    Args:
        load_data :  (str): 가져올 데이터의 주소입니다.
        tokenizer (AutoTokenizer): 데이터를 토큰화할 토크나이저입니다.

    func :
        1)preprocessing_dataset(pd.DataFrame) : 데이터 전처리를 합니다 return pd.DataFrame
        2)tokenized_dataset(pd.DataFrame, tokenizer: AutoTokenizer) : 전처리된 데이터셋을 tokenized_dataset으로 반환 합니다
        3)get_dataset : train, valid 데이터셋을 토큰화 셋으로 변환한 뒤 RE_Dataset 형태로 반환 합니다
        4)get_test_dataset : test 데이터셋을 토큰화 셋으로 변환한 뒤 RE_Dataset 형태로 반환 합니다
    """

    def __init__(self, data_path: pathlib.Path, tokenizer: AutoTokenizer):
        super().__init__(data_path, tokenizer)

    def preprocessing_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""

        def multiply_func(x, col1, col2):
            sub_entity_dict, object_entity_dict = eval(x[col1]), eval(x[col2])

            sub_word = sub_entity_dict["word"]
            sub_start_idx = int(sub_entity_dict["start_idx"])
            sub_end_idx = int(sub_entity_dict["end_idx"])
            sub_type = sub_entity_dict["type"]

            obj_word = object_entity_dict["word"]
            obj_start_idx = int(object_entity_dict["start_idx"])
            obj_end_idx = int(object_entity_dict["end_idx"])
            obj_type = object_entity_dict["type"]

            return pd.Series(
                [
                    sub_word,
                    sub_start_idx,
                    sub_end_idx,
                    sub_type,
                    obj_word,
                    obj_start_idx,
                    obj_end_idx,
                    obj_type,
                ]
            )

        ff = partial(multiply_func, col1="subject_entity", col2="object_entity")

        dataset[
            [
                "sub_word",
                "sub_start_idx",
                "sub_end_idx",
                "sub_type",
                "obj_word",
                "obj_start_idx",
                "obj_end_idx",
                "obj_type",
            ]
        ] = dataset.apply(lambda x: ff(x), axis=1)

        re_sentence_list = []

        for idx, item in dataset[:].iterrows():
            temp_sentence, sub_word, ssidx, seidx, sub_type = (
                item["sentence"],
                item["sub_word"],
                item["sub_start_idx"],
                item["sub_end_idx"],
                item["sub_type"],
            )
            obj_word, osidx, oeidx, obj_type = (
                item["obj_word"],
                item["obj_start_idx"],
                item["obj_end_idx"],
                item["obj_type"],
            )

            re_sub_word = f"@*{sub_type}*{sub_word}@"
            re_obj_word = f"#∧{obj_type}∧{obj_word}#"

            if ssidx < osidx:
                if ssidx == 0:
                    temp = "".join(
                        [
                            temp_sentence[ssidx : seidx + 1].replace(
                                sub_word, re_sub_word
                            ),
                            temp_sentence[seidx + 1 : osidx],
                            temp_sentence[osidx : oeidx + 1].replace(
                                obj_word, re_obj_word
                            ),
                            temp_sentence[oeidx + 1 :],
                        ]
                    )
                else:
                    temp = "".join(
                        [
                            temp_sentence[0:ssidx],
                            temp_sentence[ssidx : seidx + 1].replace(
                                sub_word, re_sub_word
                            ),
                            temp_sentence[seidx + 1 : osidx],
                            temp_sentence[osidx : oeidx + 1].replace(
                                obj_word, re_obj_word
                            ),
                            temp_sentence[oeidx + 1 :],
                        ]
                    )
            else:
                if osidx == 0:
                    temp = "".join(
                        [
                            temp_sentence[osidx : oeidx + 1].replace(
                                obj_word, re_obj_word
                            ),
                            temp_sentence[oeidx + 1 : ssidx],
                            temp_sentence[ssidx : seidx + 1].replace(
                                sub_word, re_sub_word
                            ),
                            temp_sentence[seidx + 1 :],
                        ]
                    )
                else:
                    temp = "".join(
                        [
                            temp_sentence[0:osidx],
                            temp_sentence[osidx : oeidx + 1].replace(
                                obj_word, re_obj_word
                            ),
                            temp_sentence[oeidx + 1 : ssidx],
                            temp_sentence[ssidx : seidx + 1].replace(
                                sub_word, re_sub_word
                            ),
                            temp_sentence[seidx + 1 :],
                        ]
                    )

            re_sentence_list.append(temp)

        dataset["sentence"] = re_sentence_list[:]

        out_dataset = pd.DataFrame(
            {
                "id": dataset["id"],
                "sentence": dataset["sentence"],
                "label": dataset["label"],
            }
        )

        return out_dataset

    def tokenized_dataset(
        self, dataset: pd.DataFrame, tokenizer: AutoTokenizer
    ) -> torch.Tensor:
        """tokenizer에 따라 sentence를 tokenizing 합니다."""
        print(dataset[:5]["sentence"])

        tokenized_sentences = tokenizer(
            list(dataset["sentence"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
        return tokenized_sentences


# --------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------


def load_dataloader(
    dataloder_type: str, data_path: pathlib.Path, tokenizer: AutoTokenizer
):
    """_summary_

    Args:
        dataloder_type (str) : 가져올 dataloder 클래스 입니다 config 확인
        load_data (pathlib.Path): 가져올 데이터의 주소입니다.
        tokenizer (AutoTokenizer): 데이터를 토큰화할 토크나이저입니다.

    Returns:
        dataloader class : dataloader_type에 맞는 class 반환 합니다
    """
    dataloader_config = {
        "BaseDataLoader": BaseDataLoader(data_path, tokenizer),
        "CustomDataLoader": CustomDataLoader(data_path, tokenizer),
        "CustomDataLoader_hs": CustomDataLoader_hs(data_path, tokenizer),
    }
    return dataloader_config[dataloder_type]


def set_tokenizer(tokenizer: AutoTokenizer):
    """Tokenzier 재정의하고 , speical 토큰을 추가 합니다.

    Args:
        tokenizer (AutoTokenizer): 데이터를 토큰화할 토크나이저입니다.

    Returns:
        tokenizer: 새롭게 정의된 Tokenizer 입니다
        new_vocab_size : speical 토큰이 추가된 vocab_size 입니다 => 모델 embedding size를 추가하기 위해 사용합니다!
    """
    add_token = ["LOC", "PER", "ORG", "DAT", "NOH", "POH", "∧"]

    new_token_count = tokenizer.add_tokens(add_token)  # 새롭게 추가된 토큰의 수 저장

    new_vocab_size = tokenizer.vocab_size + new_token_count

    return tokenizer, new_vocab_size


class CustomDataLoader_hs:
    """Custom DataLoader 입니다.

    Args:
        load_data :  (str): 가져올 데이터의 주소입니다.
        tokenizer (AutoTokenizer): 데이터를 토큰화할 토크나이저입니다.

    func :
        1)preprocessing_dataset(pd.DataFrame) : 데이터 전처리를 합니다 return pd.DataFrame
        2)tokenized_dataset(pd.DataFrame, tokenizer: AutoTokenizer) : 전처리된 데이터셋을 tokenized_dataset으로 반환 합니다
        3)get_dataset : train, valid 데이터셋을 토큰화 셋으로 변환한 뒤 RE_Dataset 형태로 반환 합니다
        4)get_test_dataset : test 데이터셋을 토큰화 셋으로 변환한 뒤 RE_Dataset 형태로 반환 합니다
    """

    def __init__(self, data_path: pathlib.Path, tokenizer: AutoTokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer

    def train_preprocessing_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""

        def multiply_func(x, col1, col2):
            sub_entity_dict, object_entity_dict = eval(x[col1]), eval(x[col2])
            sub_word = sub_entity_dict["word"]
            sub_start_idx = int(sub_entity_dict["start_idx"])
            sub_end_idx = int(sub_entity_dict["end_idx"])
            sub_type = sub_entity_dict["type"]

            obj_word = object_entity_dict["word"]
            obj_start_idx = int(object_entity_dict["start_idx"])
            obj_end_idx = int(object_entity_dict["end_idx"])
            obj_type = object_entity_dict["type"]

            return pd.Series(
                [
                    sub_word,
                    sub_start_idx,
                    sub_end_idx,
                    sub_type,
                    obj_word,
                    obj_start_idx,
                    obj_end_idx,
                    obj_type,
                ]
            )

        ff = partial(multiply_func, col1="subject_entity", col2="object_entity")
        dataset[
            [
                "sub_word",
                "sub_start_idx",
                "sub_end_idx",
                "sub_type",
                "obj_word",
                "obj_start_idx",
                "obj_end_idx",
                "obj_type",
            ]
        ] = dataset.apply(lambda x: ff(x), axis=1)
        re_sentence_list = []

        for idx, item in dataset[:].iterrows():
            temp_sentence = item["sentence"]
            sub_word, sub_type = item["sub_word"], item["sub_type"]
            obj_word, obj_type = item["obj_word"], item["obj_type"]
            sub_type = change_type(sub_type)
            obj_type = change_type(obj_type)

            temp = (
                f"대상 개체명은 {sub_word}이다.[SEP]목적 개체명은 {obj_word}이다.[SEP]대상 개체의 유형은 {sub_type}이다.[SEP]목적 개체의 유형은 {obj_type}이다.[SEP]"
                + temp_sentence
            )  # + "[SEP]" + verbalize_label(item['label'],sub_word,obj_word)
            # temp = f"대상 개체명은 {sub_word}이다.목적 개체명은 {obj_word}이다.대상 개체의 유형은 {sub_type}이다.목적 개체의 유형은 {obj_type}이다." + temp_sentence # + "[SEP]" + verbalize_label(item['label'],sub_word,obj_word)
            # print(temp)
            re_sentence_list.append(temp)

        dataset["sentence"] = re_sentence_list[:]
        # print(dataset['label'])
        print("-------------------sentence")
        print(dataset["sentence"])
        out_dataset = pd.DataFrame(
            {
                "id": dataset["id"],
                "sentence": dataset["sentence"],
                "label": dataset["label"],
            }
        )
        return out_dataset

    def test_preprocessing_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""

        def multiply_func(x, col1, col2):
            sub_entity_dict, object_entity_dict = eval(x[col1]), eval(x[col2])
            sub_word = sub_entity_dict["word"]
            sub_start_idx = int(sub_entity_dict["start_idx"])
            sub_end_idx = int(sub_entity_dict["end_idx"])
            sub_type = sub_entity_dict["type"]

            obj_word = object_entity_dict["word"]
            obj_start_idx = int(object_entity_dict["start_idx"])
            obj_end_idx = int(object_entity_dict["end_idx"])
            obj_type = object_entity_dict["type"]

            return pd.Series(
                [
                    sub_word,
                    sub_start_idx,
                    sub_end_idx,
                    sub_type,
                    obj_word,
                    obj_start_idx,
                    obj_end_idx,
                    obj_type,
                ]
            )

        ff = partial(multiply_func, col1="subject_entity", col2="object_entity")
        dataset[
            [
                "sub_word",
                "sub_start_idx",
                "sub_end_idx",
                "sub_type",
                "obj_word",
                "obj_start_idx",
                "obj_end_idx",
                "obj_type",
            ]
        ] = dataset.apply(lambda x: ff(x), axis=1)
        re_sentence_list = []

        for idx, item in dataset[:].iterrows():
            temp_sentence = item["sentence"]
            sub_word, sub_type = item["sub_word"], item["sub_type"]
            obj_word, obj_type = item["obj_word"], item["obj_type"]
            sub_type = change_type(sub_type)
            obj_type = change_type(obj_type)

            temp = (
                f"대상 개체명은 {sub_word}이다.[SEP]목적 개체명은 {obj_word}이다.[SEP]대상 개체의 유형은 {sub_type}이다.[SEP]목적 개체의 유형은 {obj_type}이다.[SEP]"
                + temp_sentence
            )
            # temp = f"대상 개체명은 {sub_word}이다.목적 개체명은 {obj_word}이다.대상 개체의 유형은 {sub_type}이다.목적 개체의 유형은 {obj_type}이다." + temp_sentence
            # print(temp)
            re_sentence_list.append(temp)

        dataset["sentence"] = re_sentence_list[:]
        # print(dataset['label'])
        print("-------------------sentence")
        print(dataset["sentence"])
        out_dataset = pd.DataFrame(
            {
                "id": dataset["id"],
                "sentence": dataset["sentence"],
                "label": dataset["label"],
            }
        )
        return out_dataset

    ### test에서는 temp에 verbalize 부분을 빼야한다...

    def tokenized_dataset(
        self, dataset: pd.DataFrame, tokenizer: AutoTokenizer
    ) -> torch.Tensor:
        """tokenizer에 따라 sentence를 tokenizing 합니다."""
        # print('@'*100)
        # print(dataset['sentence'][0])

        tokenized_sentences = tokenizer(
            list(dataset["sentence"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        # print(tokenized_sentences[0])
        return tokenized_sentences

    def get_dataset(self) -> RE_Dataset:
        """데이터셋을 Trainer에 넣을 수 있도록 처리하여 리턴합니다.

        Args:
            data_path (str): 가져올 데이터의 주소입니다.
            tokenizer (AutoTokenizer): 데이터를 토큰화할 토크나이저입니다.

        Returns:
            pd.DataFrame: _description_
        """
        pd_dataset = pd.read_csv(self.data_path)
        dataset = self.train_preprocessing_dataset(pd_dataset)

        dataset_label = utils.label_to_num(dataset["label"].values)

        # tokenizing dataset
        dataset_tokens = self.tokenized_dataset(dataset, self.tokenizer)
        print("@" * 100)
        print(dataset_tokens)
        # make dataset for pytorch.
        dataset = RE_Dataset(dataset_tokens, dataset_label)
        print(dataset)
        return dataset

    def get_valid_dataset(self) -> RE_Dataset:
        """데이터셋을 Trainer에 넣을 수 있도록 처리하여 리턴합니다.

        Args:
            data_path (str): 가져올 데이터의 주소입니다.
            tokenizer (AutoTokenizer): 데이터를 토큰화할 토크나이저입니다.

        Returns:
            pd.DataFrame: _description_
        """
        pd_dataset = pd.read_csv(self.data_path)
        dataset = self.test_preprocessing_dataset(pd_dataset)

        dataset_label = utils.label_to_num(dataset["label"].values)

        # tokenizing dataset
        dataset_tokens = self.tokenized_dataset(dataset, self.tokenizer)
        # make dataset for pytorch.
        dataset = RE_Dataset(dataset_tokens, dataset_label)
        return dataset

    def get_test_dataset(self) -> RE_Dataset:
        """데이터셋을 Trainer에 넣을 수 있도록 처리하여 리턴합니다.

        Args:
            data_path (str): 가져올 데이터의 주소입니다.
            tokenizer (AutoTokenizer): 데이터를 토큰화할 토크나이저입니다.

        Returns:
            pd.DataFrame: _description_
        """
        pd_dataset = pd.read_csv(self.data_path)
        dataset = self.test_preprocessing_dataset(pd_dataset)

        dataset_label = list(map(int, dataset["label"].values))

        dataset_id = dataset["id"]
        # tokenizing dataset
        dataset_tokens = self.tokenized_dataset(dataset, self.tokenizer)
        # make dataset for pytorch.
        dataset = RE_Dataset(dataset_tokens, dataset_label)
        return dataset_id, dataset, dataset_label
