#include <vector>
#include "knn.hpp"
#include "knn.cuh"
#include "../utils.hpp"

std::vector<at::Tensor> knn_forward_cuda(at::Tensor xyz1, at::Tensor xyz2, int k)
{
    CHECK_INPUT(xyz1);CHECK_IS_FLOAT(xyz1);
    CHECK_INPUT(xyz2);CHECK_IS_FLOAT(xyz2);
    int b = xyz1.size(0);
    int c = xyz1.size(1);
    int n = xyz1.size(2);
    int m = xyz2.size(2);
    at::Tensor dist1 = torch::ones({b, k, n}, at::device(xyz1.device()).dtype(at::ScalarType::Float))*UNDEFINE_VALUE; 
    at::Tensor dist2 = torch::ones({b, k, m}, at::device(xyz1.device()).dtype(at::ScalarType::Float))*UNDEFINE_VALUE;
    at::Tensor idx1 = torch::zeros({b, k, n}, at::device(xyz1.device()).dtype(at::ScalarType::Int));
    at::Tensor idx2 = torch::zeros({b, k, m}, at::device(xyz1.device()).dtype(at::ScalarType::Int));

    //[b,n,3],[b,m,3],float[b,n] int[b,n]
    knn_kernel(b, c, n, m, k, xyz1.data_ptr<float>(), xyz2.data_ptr<float>(),
    dist1.data_ptr<float>(), dist2.data_ptr<float>(), 
	idx1.data_ptr<int>(), idx2.data_ptr<int>());

  return {dist1, dist2, idx1, idx2};
};
//[b,n,3],[b,m,3],float[b,k,n] int[b,k,n]
std::vector<at::Tensor> knn_backward_cuda(
    at::Tensor xyz1,at::Tensor xyz2,
    at::Tensor graddist1, at::Tensor graddist2, 
    at::Tensor idx1, at::Tensor idx2)
{
    CHECK_INPUT(xyz1);CHECK_IS_FLOAT(xyz1);
    CHECK_INPUT(xyz2);CHECK_IS_FLOAT(xyz2);
    CHECK_INPUT(graddist1);CHECK_IS_FLOAT(graddist1);
    CHECK_INPUT(graddist2);CHECK_IS_FLOAT(graddist2);
    CHECK_INPUT(idx1);CHECK_IS_INT(idx1);
    CHECK_INPUT(idx2);CHECK_IS_INT(idx2);
    int b = xyz1.size(0);
    int c = xyz1.size(1);
    int n = xyz1.size(2);
    int m = xyz2.size(2);
    int k = idx1.size(1);

    at::Tensor gradxyz1 = torch::zeros({b, c, n}, at::device(xyz1.device()).dtype(at::ScalarType::Float));
    at::Tensor gradxyz2 = torch::zeros({b, c, m}, at::device(xyz1.device()).dtype(at::ScalarType::Float));
    
    knn_grad_kernel(b,c,n,m,k,xyz1.data_ptr<float>(),xyz2.data_ptr<float>(),
    gradxyz1.data_ptr<float>(),gradxyz2.data_ptr<float>(),
    graddist1.data_ptr<float>(),graddist2.data_ptr<float>(),
    idx1.data_ptr<int>(),idx2.data_ptr<int>());
  return {gradxyz1, gradxyz2};
};