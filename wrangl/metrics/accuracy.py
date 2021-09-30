from .metric import Metric


class Accuracy(Metric):

    def single_forward(self, gold, pred):
        return gold == pred
