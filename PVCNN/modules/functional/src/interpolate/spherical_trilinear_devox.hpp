#ifndef _SPHERICAL_TRILINEAR_DEVOX_HPP
#define _SPHERICAL_TRILINEAR_DEVOX_HPP

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> spherical_trilinear_devoxelize_forward(const int r,
                                                     const bool is_training,
                                                     const at::Tensor coords,
                                                     const at::Tensor features,
                                                     const at::Tensor g_inds);

at::Tensor spherical_trilinear_devoxelize_backward(const at::Tensor grad_y,
                                         const at::Tensor indices,
                                         const at::Tensor weights, const int r);

#endif
