from collections import defaultdict


class Metric:

    def compute_one(self, pred, gold):
        """Computes metrics for one example"""
        raise NotImplementedError()

    def __call__(self, pred, gold):
        metrics = defaultdict(list)
        for pi, gi in zip(pred, gold):
            m = self.compute_one(pi, gi)
            for k, v in m.items():
                metrics[k].append(v)
        return {k: sum(v)/len(v) for k, v in metrics.items()}


class Accuracy(Metric):

    def compute_one(self, pred, gold):
        return {'acc': pred == gold}


class SetF1:

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
        return {k: sum(v)/len(v) for k, v in metrics.items()}
