from pathlib import Path
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


root = Path(__file__).parent
extensions = []
for name in ('relabel', 'async', 'relabel2hop'):
    sources = [str(root / 'csrc' / (name + '.cpp'))]
    cpu = root / 'csrc/cpu' / (name + '_cpu.cpp')
    if cpu.exists() and name != 'relabel2hop':
        sources.append(str(cpu))
    cuda = root / 'csrc/cuda' / (name + '_cuda.cu')
    use_cuda = CUDA_HOME is not None and cuda.exists()
    cls = CUDAExtension if use_cuda else CppExtension
    if use_cuda:
        sources.append(str(cuda))
    extensions.append(cls('torch_geometric_autoscale._' + name,
                          sources, include_dirs=[str(root / 'csrc')],
                          define_macros=[('WITH_CUDA', None)] if use_cuda else [],
                          extra_compile_args={'cxx': ['-O3', '-fopenmp'],
                                              **({'nvcc': ['-O3', '--expt-relaxed-constexpr']} if use_cuda else {})}))

setup(name='reprise-history-backend', version='0.1.0', packages=find_packages(),
      ext_modules=extensions,
      cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)})
