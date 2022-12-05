from klue.loss import FocalLoss
from transformers import Trainer


# TODO: ADD TYPE HINT AND DOCSTRING
class FocallossTrainer(Trainer):
    # gamma, alpha를 직접 설정할 수 있도록 코드를 개선하였습니다.
    # 다만 alpha는 int값을 넣을시 gather와 관련하여 오류가 발생힙니다.
    def __init__(self, gamma: int = 2, alpha: int = None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha  # alpha는 안쓰는것을 추천한다.쓰는 순간 오류가 발생하는 이슈가 있음.

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # loss_fct = MSELoss()
        # loss = loss_fct(logits.squeeze(), labels.squeeze())
        loss = FocalLoss(gamma=self.gamma, alpha=self.alpha)(
            logits.squeeze(), labels.squeeze()
        )
        return (loss, outputs) if return_outputs else loss
