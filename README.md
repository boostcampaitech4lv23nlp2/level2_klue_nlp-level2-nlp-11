# 🚀 KLUE - level 2 대회 Project
### Relation Extraction
관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

### 활용 장비 및 재료
- GPU : v100 * 5
- 협업 툴 : Github, Notion, Wandb
- 개발 환경 : Ubuntu 18.04
- 라이브러리 : torch 1.12.0, torchmetrics 0.10.0, wandb 0.13.4, sentence-transformers 2.2.2

### Command Line Interface
```
# Train
>>> cd code
>>> python main.py -m=t

# Inference
>>> cd code
>>> python main.py -m=i
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
### 👥 Members
- 김지수 [Github](https://github.com/kuotient/kuotient)
- 김현수 [Github](https://github.com/gustn9609)
- 지상수 [Github](https://github.com/ggb04110)
- 최석훈 [Github](https://github.com/soypabloo)
- 최혜원 [Github](https://github.com/soohi0/soohi0)
