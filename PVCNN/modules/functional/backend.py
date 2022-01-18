'''
Author: your name
Date: 2020-11-20 22:28:32
LastEditTime: 2020-11-23 17:13:35
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /exp1/PVCNN/modules/functional/backend.py
'''
import os

from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
_backend = load(name='_multi_shape_pvcnn_backend',
                extra_cflags=['-O3', '-std=c++17'],
                sources=[os.path.join(_src_path,'src', f) for f in [
                    'ball_query/ball_query.cpp',
                    'ball_query/ball_query.cu',
                    'grouping/grouping.cpp',
                    'grouping/grouping.cu',
                    'interpolate/neighbor_interpolate.cpp',
                    'interpolate/neighbor_interpolate.cu',
                    'interpolate/trilinear_devox.cpp',
                    'interpolate/trilinear_devox.cu',
                    'sampling/sampling.cpp',
                    'sampling/sampling.cu',
                    'voxelization/vox.cpp',
                    'voxelization/vox.cu',
                    'interpolate/spherical_trilinear_devox.cpp',
                    'interpolate/spherical_trilinear_devox.cu',
                    'spherical_voxelization/spherical_vox.cpp',
                    'spherical_voxelization/spherical_vox.cu',
                    'spherical_ppf/ppf.cpp',
                    'spherical_ppf/ppf.cu',
                    'knn/knn.cpp',
                    'knn/knn.cu',
                    'bindings.cpp',
                ]]
                )

__all__ = ['_backend']
# import os
# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# _src_path = os.path.dirname(os.path.abspath(__file__))
# setup(
#     name='backend',
#     ext_modules=[
#         CUDAExtension(
#                 name='backend',
#                 sources=[os.path.join(_src_path,'src', f) for f in [
#                     'ball_query/ball_query.cpp',
#                     'ball_query/ball_query.cu',
#                     'grouping/grouping.cpp',
#                     'grouping/grouping.cu',
#                     'interpolate/neighbor_interpolate.cpp',
#                     'interpolate/neighbor_interpolate.cu',
#                     'interpolate/trilinear_devox.cpp',
#                     'interpolate/trilinear_devox.cu',
#                     'sampling/sampling.cpp',
#                     'sampling/sampling.cu',
#                     'voxelization/vox.cpp',
#                     'voxelization/vox.cu',
#                     'interpolate/spherical_trilinear_devox.cpp',
#                     'interpolate/spherical_trilinear_devox.cu',
#                     'spherical_voxelization/spherical_vox.cpp',
#                     'spherical_voxelization/spherical_vox.cu',
#                     'spherical_ppf/ppf.cpp',
#                     'spherical_ppf/ppf.cu',
#                     'knn/knn.cpp',
#                     'knn/knn.cu',
#                     'bindings.cpp',
#                 ]],
#                 extra_compile_args={'cxx': ['-g'],
#                                     'nvcc': ['-O2']})
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })