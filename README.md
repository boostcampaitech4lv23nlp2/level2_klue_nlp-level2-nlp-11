# KLUE - level 2 대회 Project
### Relation Extraction
관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

### Command Line Interface
```
>>> cd code
>>> python train.py
```
### Project Directories
```
Relation_Extraction
 ┣code
 ┃ ┣ klue
 ┃ ┃ ┣ dataloader.py
 ┃ ┃ ┣ loss.py
 ┃ ┃ ┣ metric.py
 ┃ ┃ ┣ trainer.py
 ┃ ┃ ┗ utils.py
 ┃ ┣ inference.py
 ┃ ┣ main.py
 ┃ ┗ train.py
 ┣ config
 ┃ ┣ base_config.yaml
 ┃ ┣ config.yaml.example
 ┃ ┣ sweep.yaml
 ┃ ┗ sweep.yaml.example
 ┣ dataset
 ┃ ┣ best_model
 ┃ ┣ prediction
 ┃ ┣ results
 ┃ ┣ test
 ┃ ┣ train
 ┃ ┗ wandb
 ┣ document
 ┃ ┗ PR_manual.md
 ┣ notebook
 ┃ ┣ EDA.ipynb
 ┃ ┣ false_alarm.ipynb
 ┃ ┗ train_valid_split.ipynb
 ┣ .gitignore
 ┣ README.md
 ┗ format.sh
```
### Members

### Auto Format
먼저 pip를 통해 관련 라이브러리를 설치합니다.
```
pip install black
pip install isort
```
이후 터미널에서 다음 명령어를 통해 파일을 formatting할수 있습니다.

```sh format.sh```

### wandb experiment rule
wandb실험을 효율적으로 하기 위한 룰입니다!
1. base_config.wandb.exp_name 으로 본인의 실험 목적을 설정합니다.(ex: 기본 loss와 focalloss와의 차이를 비교하는 실험 -> Compare_loss)
2. base_config.wandb.annotation 으로 세부실험의 주요 변경사항을 표시합니다.(ex:배치사이즈를 바꾸었다. -> change_batch)
3. 실험 각각의 이름은 "lr-(learning_rate값)_(2. 에서 설정한 annotation)"의 형태로 설정되어 있습니다.
4. 기본적으로 모델의 이름으로 그룹이 나뉘어져 있습니다.(모델별로 비교하기 용이하게 하기 위함.)

### Wandb sweep
기본적인 sweep설명을 하면, sweep.yaml파일을 만들고(example 파일을 추가하였으니 확인해주세요.),
wandb sweep sweep.yaml을 입력하여 sweep을 만듭니다.
이후wandb agent [AGENT ID]를 입력하면 sweep이 실행됩니다.
기본적인 sweep로직은 다음과 같습니다.
1. base_config.yaml파일에서 모든 config를 가져옵니다.
2. 이후 sweep이 cli argument를 이용하여 sweep할 parameter를 가져와 기존 config에서 갱신합니다.
#### 주의사항1: sweep.yaml의 program을 본인의 파일 경로에 맞게 설정해주세요.(sweep을 여러명이서 하려면 공통된 파일경로 설정이 필요해서 개선해야할 사항입니다.)
#### 주의사항2: base_config.yaml설정이 동일하지 않으면 sweep을 여러 서버에서 실행했을 때, 잘못된 결과가 나올 수 있습니다.
ex) 서버1의 base_config에는 모델이 klue-roberta-large이고, 서버2의 base_config에는 klue-roberta-small일 경우, sweep에서 모델을 따로 설정해 주지 않으면 잘못된 결과가 나옵니다!

#### 주의사항3: sweep.yaml에 project와 name설정을 해주세요! sweep은 base_config의 project와 name을 무시합니다.