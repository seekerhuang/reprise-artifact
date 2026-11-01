import torch
import math
from main.products_gcn.products_common_bn_mean_p2_v1 import CommonBNMeanP2

class BNStructuralMeanP2(CommonBNMeanP2):

    def __init__(self, *args, strength=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(strength) or strength < 0:
            raise ValueError('Finite nonnegative structural strength required')
        if any((group.get('weight_decay', 0.0) != 0.0 for group in self.optimizer.param_groups)):
            raise ValueError('This declared candidate preserves the WD=0 recipe')
        self.strength = float(strength)
        self.stat = torch.zeros(7, dtype=torch.float64, device=self.features.device)
        self.last_measurement = None
        self._stat_one = self.stat.new_ones(())
