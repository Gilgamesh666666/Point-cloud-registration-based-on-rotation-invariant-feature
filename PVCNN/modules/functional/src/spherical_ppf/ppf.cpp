#include "ppf.hpp"
#include "ppf.cuh"

#include "../utils.hpp"

/*
  Function: average pool voxelization (forward)
  Args:
    coords : coords of each point, FloatTensor[b, 3, n] 原始coords
    center : center point, FloatTensor[b, 3, n]
    normals : normals of each point, FloatTensor[b, 3, n] 原始coords
    center_normal : normal of center point, FloatTensor[b, 3, n]
  output:
    feat: features, FloatTensor[b, 4, n]
*/

at::Tensor spherical_ppf_forward(const at::Tensor coords,
                                             const at::Tensor center,
                                             const at::Tensor normals,
                                             const at::Tensor center_normal) {
  CHECK_CUDA(coords);CHECK_CONTIGUOUS(coords);CHECK_IS_FLOAT(coords);
  CHECK_CUDA(center);CHECK_CONTIGUOUS(center);CHECK_IS_FLOAT(center);
  CHECK_CUDA(normals);CHECK_CONTIGUOUS(normals);CHECK_IS_FLOAT(normals);
  CHECK_CUDA(center_normal);CHECK_CONTIGUOUS(center_normal);CHECK_IS_FLOAT(center_normal);
  
  int b = coords.size(0);
  int n = coords.size(2);
  //printf("%d", n);
  at::Tensor feat = torch::zeros(
      {b, 4, n}, at::device(coords.device()).dtype(at::ScalarType::Float));
  
  spherical_ppf(b, n, coords.data_ptr<float>(),
               center.data_ptr<float>(), normals.data_ptr<float>(),
               center_normal.data_ptr<float>(), feat.data_ptr<float>());
  return feat;
}