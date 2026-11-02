import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


def main():
    if CUDA_HOME is None:
        raise RuntimeError('CUDA toolkit with nvcc is required')
    root = Path(__file__).resolve().parent
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation',
                    str(root / 'empirical/vendor/gas')], check=True)
    subprocess.run(['cc', '-O3', '-shared', '-fPIC',
                    str(root / 'main/sage/sage_direct_self_loop_csc_v1.c'),
                    '-o', str(root / 'main/sage/libsage_direct_self_loop_csc_v1.so')], check=True)
    source = root / 'main/native'
    build = source / 'build'
    build.mkdir(exist_ok=True)
    library = load(name='reprise_dgl_tensoradapter', sources=[str(source / 'torch_adapter.cpp')],
                   extra_include_paths=[str(source), str(Path(CUDA_HOME) / 'include')],
                   extra_cflags=['-O3', '-DDGL_USE_CUDA'],
                   extra_ldflags=['-lc10_cuda', '-ltorch_cuda', '-L' + str(Path(CUDA_HOME) / 'lib64'), '-lcudart'],
                   build_directory=str(build), is_python_module=False, verbose=True)
    target = source / 'lib/tensoradapter/pytorch'
    target.mkdir(parents=True, exist_ok=True)
    version = torch.__version__.split('+')[0]
    shutil.copy2(library, target / ('libtensoradapter_pytorch_' + version + '.so'))
    spec = importlib.util.find_spec('dgl')
    dgl_root = Path(next(iter(spec.submodule_search_locations)))
    link = source / 'lib/libdgl.so'
    if link.is_symlink():
        link.unlink()
    link.symlink_to(dgl_root / 'libdgl.so')
    helpers = root / 'empirical/saint'
    inputs = helpers / 'input'
    head = helpers / 'head'
    subprocess.run(['cc', '-O3', '-shared', '-fPIC', str(inputs / 'binary_labels_v1.c'),
                    '-o', str(inputs / 'libbinary_labels_v1.so')], check=True)
    subprocess.run([str(Path(CUDA_HOME) / 'bin/nvcc'), '-O3', '--fmad=false', '-shared',
                    '-Xcompiler=-fPIC', '-gencode=arch=compute_80,code=sm_80',
                    str(head / 'mu_elementwise.cu'), '-o', str(head / 'libmu_elementwise_sm80.so')], check=True)
    subprocess.run(['cc', '-O3', '-shared', '-fPIC', str(inputs / 'reddit_bank_histogram_cpu_v1.c'),
                    '-o', str(inputs / 'libreddit_bank_histogram_cpu_v1.so')], check=True)
    subprocess.run([str(Path(CUDA_HOME) / 'bin/nvcc'), '-O3', '-std=c++17', '--shared',
                    '-Xcompiler=-fPIC', '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true',
                    '-gencode=arch=compute_80,code=sm_80', str(head / 'reddit_half_pointwise_v1.cu'),
                    '-o', str(head / 'libreddit_half_pointwise_sm80.so')], check=True)
    workspace_build = head / 'build'
    workspace_build.mkdir(exist_ok=True)
    workspace = load(name='reprise_serial_workspace', sources=[str(head / 'serial_blas_workspace_owner_v1.cpp')],
                     extra_include_paths=[str(Path(CUDA_HOME) / 'include')],
                     extra_cflags=['-O3'], extra_ldflags=['-lc10_cuda', '-ltorch_cuda'],
                     build_directory=str(workspace_build), is_python_module=False, verbose=True)
    shutil.copy2(workspace, head / 'libserial_blas_workspace_owner_v1.so')
    saint = root / 'empirical/vendor/graphsaint'
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=saint, check=True)


if __name__ == '__main__':
    main()
