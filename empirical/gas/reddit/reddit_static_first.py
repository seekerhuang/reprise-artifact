from empirical.gas.reddit.reddit_supervised import Training as PreviousTraining
import time
import torch

class Training(PreviousTraining):

    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        self.cached_first_queries = 0
        self.cached_first_rows = 0
