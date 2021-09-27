from .metric import Metric
from typing import List, Dict, Tuple


class Accuracy(Metric):

    def __init__(self, key: str):
        """
        Args:
            key: key to compute accuracy on.
        """
        self.key = key

    def forward(self, preds: List[Tuple[Dict, Dict]]) -> dict:
        acc = total = 0
        for ex, pred in preds:
            g = ex[self.key]
            p = pred[self.key]
            acc += g.eq(p).sum()
            total += g.numel()
        return dict(accuracy=(acc/total).item())

    def should_early_stop(self, score, best_score):
        return super().should_early_stop(score, best_score, larger_is_better=True)
