from main.products_gcn.products_graph_decode_v1 import GraphDecode

class GraphMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bn = self.model.bns[1]
        self.recovery_graph = GraphDecode(selected=(bn.weight, bn.bias), strength=self.strength)

    def verify(self):
        clocks = super().verify()
        work = self.recovery_graph.work()
        if (work['calls'], work['capture_count'], work['warmup_calculations'], work['staging_copies']) != (self.blocks, 1, 3, 4 * self.blocks):
            raise RuntimeError('Every original recovery block must use the one paid Graph')
        self.recovery_graph.close()
        return clocks
