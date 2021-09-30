from typing import List, Tuple


class Metric:

    def single_forward(self, gold, pred):
        """
        Computes score given gold and prediction
        """
        raise NotImplementedError()

    def forward(self, preds: List[tuple]) -> dict:
        """
        Computes score given list of prediction tuples `(gold, pred)`.

        Returns:
            a dictionary entry of name to score.
        """
        score = total = 0
        for gold, pred in preds:
            score += self.single_forward(gold, pred)
            total += 1
        return {self.__class__.__name__.lower(): score/total}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def extract_key_metric(self, scores: dict) -> Tuple[str, float]:
        """
        Args:
            scores: score dictionary.
        Returns:
            key metric name and value.
        """
        m = self.__class__.__name__.lower()
        return m, scores[m]

    def should_early_stop(self, score: float, best_score: float, larger_is_better: bool = True) -> bool:
        """
        Args:
            score: current score.
            best_score: previous best score.
            larger_is_better: whether a larger score is better.

        Return:
            whether a new best scores has been found.
        """
        if best_score is None:
            return True
        if larger_is_better:
            return score >= best_score
        else:
            return score <= best_score
