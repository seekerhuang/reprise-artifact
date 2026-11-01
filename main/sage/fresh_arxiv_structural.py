from main.sage.sampled_mean_p2 import SampledMeanP2
from main.sage.sage_structural import SampledSAGEStructural

class FreshArxivStructural(SampledMeanP2):

    def __init__(self, *args, observe=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.observe = observe
        self.loss_weights = None
        self.measurements = []
        self.work.update(structural_measurements=0, structural_installs=0, extra_prefix_forwards=0, extra_prefix_VJPs=0, extra_graph_multiplications=0, optimizer_API_calls=0)
    geometry = SampledSAGEStructural.geometry
    head_measurement = SampledSAGEStructural.head_measurement
    update_block = SampledSAGEStructural.update_block
