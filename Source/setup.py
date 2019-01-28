from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='Sampling',
    ext_modules=[
        CUDAExtension(
            'Sampling_gpu',
            ['Sampling/Sampling.cu']
        )

    ],
    cmdclass={'build_ext': BuildExtension}
)
