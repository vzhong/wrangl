from .metric import Metric


class Precision(Metric):

    @classmethod
    def single_forward(cls, gold: set, pred: set):
        if not pred:
            return 0
        return len(pred.intersection(gold)) / len(pred)


class Recall(Metric):

    @classmethod
    def single_forward(cls, gold: set, pred: set):
        if not gold:
            return 0
        return len(pred.intersection(gold)) / len(gold)


class F1Score(Metric):

    @classmethod
    def single_forward(cls, gold: set, pred: set):
        precision = Precision.single_forward(gold, pred)
        recall = Recall.single_forward(gold, pred)
        denom = precision + recall
        return 2 * precision * recall / denom if denom > 0 else 0
