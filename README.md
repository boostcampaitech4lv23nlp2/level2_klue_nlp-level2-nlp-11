# π KLUE - level 2 λν Project
### Relation Extraction
κ΄κ³ μΆμΆ(Relation Extraction)μ λ¬Έμ₯μ λ¨μ΄(Entity)μ λν μμ±κ³Ό κ΄κ³λ₯Ό μμΈ‘νλ λ¬Έμ μλλ€. κ΄κ³ μΆμΆμ μ§μ κ·Έλν κ΅¬μΆμ μν ν΅μ¬ κ΅¬μ± μμλ‘, κ΅¬μ‘°νλ κ²μ, κ°μ  λΆμ, μ§λ¬Έ λ΅λ³νκΈ°, μμ½κ³Ό κ°μ μμ°μ΄μ²λ¦¬ μμ© νλ‘κ·Έλ¨μμ μ€μν©λλ€. λΉκ΅¬μ‘°μ μΈ μμ°μ΄ λ¬Έμ₯μμ κ΅¬μ‘°μ μΈ tripleμ μΆμΆν΄ μ λ³΄λ₯Ό μμ½νκ³ , μ€μν μ±λΆμ ν΅μ¬μ μΌλ‘ νμν  μ μμ΅λλ€.

### νμ© μ₯λΉ λ° μ¬λ£
- GPU : v100 * 5
- νμ ν΄ : Github, Notion, Wandb
- κ°λ° νκ²½ : Ubuntu 18.04

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
 β£code
 β β£ klue
 β β β£ dataloader.py
 β β β£ loss.py
 β β β£ metric.py
 β β β£ trainer.py
 β β β utils.py
 β β£ inference.py
 β β£ main.py
 β β train.py
 β£ config
 β β£ base_config.yaml
 β β£ config.yaml.example
 β β£ sweep.yaml
 β β sweep.yaml.example
 β£ dataset
 β β£ best_model
 β β£ prediction
 β β£ results
 β β£ test
 β β£ train
 β β wandb
 β£ document
 β β PR_manual.md
 β£ notebook
 β β£ EDA.ipynb
 β β£ false_alarm.ipynb
 β β train_valid_split.ipynb
 β£ .gitignore
 β£ README.md
 β format.sh
```
### π₯ Members
- κΉμ§μ [Github](https://github.com/kuotient/kuotient)
- κΉνμ [Github](https://github.com/gustn9609)
- μ§μμ [Github](https://github.com/ggb04110)
- μ΅μν [Github](https://github.com/soypabloo)
- μ΅νμ [Github](https://github.com/soohi0/soohi0)
