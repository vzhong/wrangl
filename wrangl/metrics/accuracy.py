from .metric import Metric
from typing import List, Dict, Tuple


class Accuracy(Metric):

    def forward(self, preds: List[Tuple[Dict, Dict]]) -> dict:
        acc = total = 0
        for g, p in preds:
            acc += g == p
            total += 1
        return dict(accuracy=(acc/total).item())

    def should_early_stop(self, score, best_score):
        return super().should_early_stop(score, best_score, larger_is_better=True)
