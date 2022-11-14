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
├── code
│   ├── klue
│   │   ├── dataloader.py
│   │   ├── metric.py
│   │   └── utils.py
│   ├── args.py
│   ├── inference.py
│   ├── train.py
│   └── requirements.txt
├── dataset
│   ├── test
│   ├── train
│   ├── best_model
│   ├── logs
│   ├── prediction
│   └── results
└── notebooks
```
### Members

### Auto Format
먼저 pip를 통해 관련 라이브러리를 설치합니다.
```pip install black```
```pip install isort```
이후 터미널에서 다음 명령어를 통해 파일을 formatting할수 있습니다.

```sh format.sh```