from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Tree learning',
  ext_modules = cythonize("trees.pyx"),
)

