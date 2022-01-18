#ifndef _PPF_HPP
#define _PPF_HPP

#include <torch/torch.h>
#include <vector>

at::Tensor spherical_ppf_forward(const at::Tensor coords,
                                const at::Tensor center,
                                const at::Tensor normals,
                                const at::Tensor center_normal);

#endif
