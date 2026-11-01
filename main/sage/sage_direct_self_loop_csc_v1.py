from pathlib import Path
import ctypes
VERSION = 'SAGE-direct-original-CSC-self-loop-layout-v1'
HERE = Path(__file__).resolve().parent
LIBRARY = HERE / 'libsage_direct_self_loop_csc_v1.so'
_LIB = None

def native_function():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(LIBRARY))
        function = _LIB.sage_append_self_loops_csc_i64_i32
        function.argtypes = [ctypes.c_int64, ctypes.c_int64] + [ctypes.c_void_p] * 5
        function.restype = ctypes.c_int
    return _LIB.sage_append_self_loops_csc_i64_i32

def candidate(dgl, graph):
    import torch
    (n, e) = (graph.num_nodes(), graph.num_edges())
    if graph.device.type != 'cpu' or graph.idtype != torch.int64 or (not graph.is_homogeneous) or graph.is_block or graph.ndata or graph.edata or (n + e >= 2 ** 31 - 1) or (graph.formats() != {'created': ['csc'], 'not created': ['coo', 'csr']}):
        raise ValueError('Original empty-feature CPU CSC64 graph required')
    (ptr, idx, eid) = graph.adj_tensors('csc')
    if eid.numel() or not ptr.is_contiguous() or (not idx.is_contiguous()):
        raise ValueError('Original implicit CSC edge IDs and contiguous arrays required')
    out_ptr = torch.empty(n + 1, dtype=torch.int32)
    out_idx = torch.empty(e + n, dtype=torch.int32)
    out_eid = torch.empty(e + n, dtype=torch.int32)
    result = native_function()(n, e, ptr.data_ptr(), idx.data_ptr(), out_ptr.data_ptr(), out_idx.data_ptr(), out_eid.data_ptr())
    if result:
        raise ValueError('Invalid original CSC data: native error ' + str(result))
    final = dgl.graph(('csc', (out_ptr, out_idx, out_eid)), num_nodes=n, idtype=torch.int32, device='cpu').formats(['csc'])
    assert final.num_nodes() == n and final.num_edges() == e + n
    return final
