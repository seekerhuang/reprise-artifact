from empirical.gas.reddit.reddit_regularization import Training as PreviousTraining
import time
import torch

class Training(PreviousTraining):

    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        self.supervised_rows = None
        self.original_last_rows = 0
        self.executed_last_rows = 0
