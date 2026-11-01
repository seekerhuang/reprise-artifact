from empirical.gas.reddit.reddit_postcommit import Training as PreviousTraining
import time
import torch

class Training(PreviousTraining):

    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        self.next_prefix_before_refresh = None
        self.completed_input_intervals = 0
        self.omitted_dead_writes = 0

    def correct_core(self, core, args):
        if self.next_prefix_before_refresh is True:
            super().correct_core(core, args)
        elif self.next_prefix_before_refresh is False:
            self.omitted_dead_writes += 1
        else:
            raise RuntimeError('Caller must declare whether an actual prefix precedes full refresh')
