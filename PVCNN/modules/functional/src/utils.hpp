/*
 * @Author: your name
 * @Date: 2020-11-20 22:28:32
 * @LastEditTime: 2020-11-21 11:56:25
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /exp1/PVCNN/modules/functional/src/utils.hpp
 */
#ifndef _UTILS_HPP
#define _UTILS_HPP

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
//.type().is_cuda()
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor") \

#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)                     \

#define CHECK_IS_INT(x)                                                        \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Int,                             \
           #x " must be an int tensor")

#define CHECK_IS_FLOAT(x)                                                      \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float,                           \
           #x " must be a float tensor")

#endif
