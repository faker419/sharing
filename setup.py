from setuptools import setup, Extension
from pybind11.setup_helpers import build_ext
import pybind11

ext_modules = [
    Extension(
        'CreateFunctionsFinal9',  # Ensure the name matches the extension
        ['create_functions.cpp'],  # Your C++ source file
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-fopenmp', '-std=c++11'],  # Add OpenMP flag
    ),
]

setup(
    name='CreateFunctionsFinal9',  # Ensure the name matches here as well
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},  # Use pybind11's build_ext helper
)
