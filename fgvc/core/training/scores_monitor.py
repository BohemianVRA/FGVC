from typing import Callable


class ScoresMonitor:
    def __init__(self, metrics_fc: Callable, num_samples: int, num_batches: int, store_preds_targs: bool = False):
        self.metrics_fc = metrics_fc
        self.num_samples = num_samples
        self.num_batches = num_batches
        self.store_preds_targs = store_preds_targs

        # set variables with information to aggregate
        self.avg_loss = None
        self.avg_scores = None
        self.preds = None
        self.targs = None

    def reset(self):
        self.loss = None
        self.avg_scores = None
        self.preds = None
        self.targs = None

    def update(self, preds, targs, loss):
        # aggregate loss
        self.avg_loss += loss / self.num_batches

        # compute scores
        batch_scores = self.metrics_fc(preds, targs)
        batch_scores = {k: v / self.num_samples for k, v in batch_scores.items()}
        if self.avg_scores is None:
            self.avg_scores = batch_scores
        else:
            for k in self.avg_scores.keys():
                self.avg_scores[k] += batch_scores[k]

        # store preds and targs

