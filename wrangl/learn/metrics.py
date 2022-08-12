from collections import defaultdict


class Metric:
    """
    Interface for a metric.
    """

    def compute_one(self, pred, gold):
        """
        Computes metrics for one example.
        You must implement this.

        Args:
            pred: single prediction.
            gold: corresponding ground truth.
        """
        raise NotImplementedError()

    def __call__(self, pred, gold):
        return self.forward(pred, gold)

    def forward(self, pred: list, gold: list) -> dict:
        """
        Computes metric over list of predictions and ground truths and returns a dictionary of scores.

        Args:
            pred: list of predictions.
            gold: corresponding ground truths.
        """
        metrics = defaultdict(list)
        for pi, gi in zip(pred, gold):
            m = self.compute_one(pi, gi)
            for k, v in m.items():
                metrics[k].append(v)
        return {k: sum(v)/len(v) for k, v in metrics.items()}


class Accuracy(Metric):
    """
    Computes exact match accuracy under the key "acc".
    """

    def compute_one(self, pred, gold):
        return {'acc': pred == gold}


class MSE(Metric):
    """
    Computes mean squared error under the key "mse".
    """

    def compute_one(self, pred, gold):
        return {'mse': (pred - gold) ** 2}


class SetF1(Metric):
    """
    Computes F1 score under the key "f1", "precision", and "recall".
    Here, both single prediction and ground truth are assumed to be a `set`.
    """

    def compute_one(self, pred: set, gold: set):
        common = pred.intersection(gold)
        precision = len(common) / max(1, len(pred))
        recall = len(common) / max(1, len(gold))
        denom = precision + recall
        f1 = (precision * recall * 2 / denom) if denom > 0 else 0
        return dict(f1=f1, precision=precision, recall=recall)

    def __call__(self, pred, gold, ignore_empty=False):
        metrics = dict(f1=[], recall=[], precision=[])
        for pi, gi in zip(pred, gold):
            if ignore_empty and (not gi or gi == {''}):
                continue
            for k, v in self.compute_one(pi, gi).items():
                metrics[k].append(v)
        return {k: (sum(v)/len(v)) for k, v in metrics.items()}


class Rouge(Metric):
    """
    Computes Rouge score under the key "rouge".
    Here, both single prediction and ground truth are assumed to be a `str`.
    You must install `rouge_scorer` for this to work.
    """

    def __init__(self):
        from rouge_score import rouge_scorer
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def compute_one(self, pred: str, gold: str):
        return dict(rouge=self.scorer.score(pred, gold)['rougeL'].fmeasure)
