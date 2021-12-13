class Accuracy:

    def compute_one(self, pred, gold):
        return pred == gold

    def __call__(self, pred, gold):
        metrics = dict(acc=[])
        for pi, gi in zip(pred, gold):
            metrics['acc'].append(self.compute_one(pi, gi))
        return {k: sum(v)/len(v) for k, v in metrics.items()}


class SetF1:

    def compute_one(self, pred: set, gold: set):
        common = pred.intersection(gold)
        precision = len(common) / max(1, len(pred))
        recall = len(common) / max(1, len(gold))
        f1 = precision * recall * 2 / max(1, precision + recall)
        return dict(f1=f1, precision=precision, recall=recall)

    def __call__(self, pred, gold, ignore_empty=False):
        metrics = dict(f1=[], recall=[], precision=[])
        for pi, gi in zip(pred, gold):
            if ignore_empty and (not gi or gi == {''}):
                continue
            for k, v in self.compute_one(pi, gi).items():
                metrics[k].append(v)
        return {k: sum(v)/len(v) for k, v in metrics.items()}
