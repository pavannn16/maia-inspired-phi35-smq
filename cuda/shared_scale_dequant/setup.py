from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="shared_scale_dequant",
    ext_modules=[
        CUDAExtension(
            name="shared_scale_dequant_cuda",
            sources=["shared_scale_dequant.cpp", "shared_scale_dequant_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                # sm_80 = A100; compatible with sm_80 and forward via PTX
                "nvcc": ["-O3", "-arch=sm_80"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
