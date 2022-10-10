from typing import Callable


class ScoresMonitor:
    def __init__(self, metrics_fc: Callable, num_samples: int):
        self.metrics_fc = metrics_fc
        self.num_samples = num_samples
        self.avg_scores = None

    def reset(self):
        self.avg_scores = None

    def update(self, preds, targs):
        batch_scores = self.metrics_fc(preds, targs)
        batch_scores = {k: v / self.num_samples for k, v in batch_scores.items()}
        if self.avg_scores is None:
            self.avg_scores = batch_scores
        else:
            for k in self.avg_scores.keys():
                self.avg_scores[k] += batch_scores[k]
