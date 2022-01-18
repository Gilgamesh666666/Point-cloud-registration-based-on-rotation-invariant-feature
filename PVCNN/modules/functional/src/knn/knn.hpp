#ifndef _KNN_HPP
#define _KNN_HPP
#include<torch/extension.h>
#include<vector>

std::vector<at::Tensor> knn_forward_cuda(at::Tensor xyz1, at::Tensor xyz2, int k);
std::vector<at::Tensor> knn_backward_cuda(
    at::Tensor xyz1,at::Tensor xyz2,
    at::Tensor graddist1, at::Tensor graddist2, 
    at::Tensor idx1, at::Tensor idx2);

#endif