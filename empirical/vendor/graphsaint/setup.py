from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

modules = [Extension("graphsaint." + name, ["graphsaint/" + name + ".pyx"], language="c++", include_dirs=[numpy.get_include()], extra_compile_args=["-O3", "-fopenmp"], extra_link_args=["-fopenmp"]) for name in ("cython_utils", "cython_sampler", "norm_aggr")]
setup(name="reprise-saint-sampler", version="1.0", packages=["graphsaint"], ext_modules=cythonize(modules, language_level=3))
