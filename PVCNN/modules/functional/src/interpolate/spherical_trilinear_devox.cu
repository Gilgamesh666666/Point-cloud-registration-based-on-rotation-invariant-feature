#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../cuda_utils.cuh"
#define PI acos(-1.0)
/*
  Function: spherical_trilinear devoxlization (forward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    r   : voxel resolution
    r2  : r ** 2
    r3  : r ** 3
    //这是连续的坐标，不是grid坐标
    coords : the coordinates of points, FloatTensor[b, 3, n]
    feat   : features, FloatTensor[b, c, r3]
    g_inds : the voxel indices of point IntTensor[b, n]
    inds   : the voxel indices of point cube, IntTensor[b, 8, n]
    wgts   : weight for trilinear interpolation, FloatTensor[b, 8, n]
    outs   : outputs, FloatTensor[b, c, n]
*/
__global__ void spherical_trilinear_devoxelize_kernel(int b, int c, int n, int r, int r2,
                                            int r3, bool is_training,
                                            const float *__restrict__ coords,
                                            const float *__restrict__ feat,
                                            const int *__restrict__ g_inds,
                                            int *__restrict__ inds,
                                            float *__restrict__ wgts,
                                            float *__restrict__ outs) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index * n * 3;
  g_inds += batch_index * n;
  inds += batch_index * n * 8;
  wgts += batch_index * n * 8;
  feat += batch_index * c * r3;
  outs += batch_index * c * n;

  for (int i = index; i < n; i += stride) {
    int pos = g_inds[i];
    if(pos==-1)
    {
      inds[i] = -1;
      continue;
    }
    float x = coords[i];
    float y = coords[i + n];
    float z = coords[i + n + n];
    //得到点的球坐标
    float gama = sqrt(x * x + y * y + z * z);
    float alpha, beta;
    if((gama==0)||(gama>=1)||((z / gama) > 1)||((z / gama) < -1))continue;
    else
    {
      
      beta = acos(z / gama);
      if(beta>=PI)continue;
      if(x==0&&y!=0) alpha = (y/abs(y))*PI*0.5;
      else if(x==0&&y==0) alpha = 0;
      else alpha = atan(y/x) + PI*(1-(x/abs(x)))/2;
      alpha += PI/r;
      if(alpha<0)alpha += 2*PI;
    }
    
    int grid_gama = pos/r2;
    int grid_alpha = (pos - grid_gama*r2)/r;
    int grid_beta = pos - grid_gama*r2 - grid_alpha*r;

    float gama_lo_f = grid_gama / r;
    float alpha_lo_f = PI*2*grid_alpha / r; 
    float beta_lo_f = PI*grid_beta / r;

    float gama_d_1 = gama - gama_lo_f; // / (x_hi_f - x_lo_f + 1e-8f)
    float alpha_d_1 = alpha - alpha_lo_f;
    float beta_d_1 = beta - beta_lo_f;
    float gama_d_0 = 1.0f - gama_d_1;
    float alpha_d_0 = 1.0f - alpha_d_1;
    float beta_d_0 = 1.0f - beta_d_1;

    float wgt000 = gama_d_0 * alpha_d_0 * beta_d_0;
    float wgt001 = gama_d_0 * alpha_d_0 * beta_d_1;
    float wgt010 = gama_d_0 * alpha_d_1 * beta_d_0;
    float wgt011 = gama_d_0 * alpha_d_1 * beta_d_1;
    float wgt100 = gama_d_1 * alpha_d_0 * beta_d_0;
    float wgt101 = gama_d_1 * alpha_d_0 * beta_d_1;
    float wgt110 = gama_d_1 * alpha_d_1 * beta_d_0;
    float wgt111 = gama_d_1 * alpha_d_1 * beta_d_1;

    int gama_lo = static_cast<int>(gama_lo_f);
    int alpha_lo = static_cast<int>(alpha_lo_f);
    int beta_lo = static_cast<int>(beta_lo_f);
    int gama_hi = (gama_d_1 > 0) ? -1 : 0;
    int alpha_hi = (alpha_d_1 > 0) ? -1 : 0;
    int beta_hi = (beta_d_1 > 0) ? 1 : 0;

    int idx000 = gama_lo * r2 + alpha_lo * r + beta_lo;
    int idx001 = idx000 + beta_hi;      // x_lo * r2 + y_lo * r + z_hi;
    int idx010 = idx000 + (alpha_hi & r);  // x_lo * r2 + y_hi * r + z_lo;
    int idx011 = idx010 + beta_hi;      // x_lo * r2 + y_hi * r + z_hi;
    int idx100 = idx000 + (gama_hi & r2); // x_hi * r2 + y_lo * r + z_lo;
    int idx101 = idx100 + beta_hi;      // x_hi * r2 + y_lo * r + z_hi;
    int idx110 = idx100 + (alpha_hi & r);  // x_hi * r2 + y_hi * r + z_lo;
    int idx111 = idx110 + beta_hi;      // x_hi * r2 + y_hi * r + z_hi;

    //if (is_training) {
      wgts[i] = wgt000;
      wgts[i + n] = wgt001;
      wgts[i + n * 2] = wgt010;
      wgts[i + n * 3] = wgt011;
      wgts[i + n * 4] = wgt100;
      wgts[i + n * 5] = wgt101;
      wgts[i + n * 6] = wgt110;
      wgts[i + n * 7] = wgt111;
      
      inds[i] = idx000;
      inds[i + n] = idx001;
      inds[i + n * 2] = idx010;
      inds[i + n * 3] = idx011;
      inds[i + n * 4] = idx100;
      inds[i + n * 5] = idx101;
      inds[i + n * 6] = idx110;
      inds[i + n * 7] = idx111;
    //}

    for (int j = 0; j < c; j++) {
      int jr3 = j * r3;
      outs[j * n + i] =
          wgt000 * feat[jr3 + idx000] + wgt001 * feat[jr3 + idx001] +
          wgt010 * feat[jr3 + idx010] + wgt011 * feat[jr3 + idx011] +
          wgt100 * feat[jr3 + idx100] + wgt101 * feat[jr3 + idx101] +
          wgt110 * feat[jr3 + idx110] + wgt111 * feat[jr3 + idx111];
    }
  }
}

/*
  Function: trilinear devoxlization (backward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    r3  : voxel cube size = voxel resolution ** 3
    inds   : the voxel indices of point cube, IntTensor[b, 8, n]
    wgts   : weight for trilinear interpolation, FloatTensor[b, 8, n]
    grad_y : grad outputs, FloatTensor[b, c, n]
    grad_x : grad inputs, FloatTensor[b, c, r3]
*/
__global__ void spherical_trilinear_devoxelize_grad_kernel(
    int b, int c, int n, int r3, const int *__restrict__ inds,
    const float *__restrict__ wgts, const float *__restrict__ grad_y,
    float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  inds += batch_index * n * 8;
  wgts += batch_index * n * 8;
  grad_x += batch_index * c * r3;
  grad_y += batch_index * c * n;

  for (int i = index; i < n; i += stride) {
    int idx000 = inds[i];
    if(idx000==-1)continue;
    int idx001 = inds[i + n];
    int idx010 = inds[i + n * 2];
    int idx011 = inds[i + n * 3];
    int idx100 = inds[i + n * 4];
    int idx101 = inds[i + n * 5];
    int idx110 = inds[i + n * 6];
    int idx111 = inds[i + n * 7];
    float wgt000 = wgts[i];
    float wgt001 = wgts[i + n];
    float wgt010 = wgts[i + n * 2];
    float wgt011 = wgts[i + n * 3];
    float wgt100 = wgts[i + n * 4];
    float wgt101 = wgts[i + n * 5];
    float wgt110 = wgts[i + n * 6];
    float wgt111 = wgts[i + n * 7];

    for (int j = 0; j < c; j++) {
      int jr3 = j * r3;
      float g = grad_y[j * n + i];
      atomicAdd(grad_x + jr3 + idx000, wgt000 * g);
      atomicAdd(grad_x + jr3 + idx001, wgt001 * g);
      atomicAdd(grad_x + jr3 + idx010, wgt010 * g);
      atomicAdd(grad_x + jr3 + idx011, wgt011 * g);
      atomicAdd(grad_x + jr3 + idx100, wgt100 * g);
      atomicAdd(grad_x + jr3 + idx101, wgt101 * g);
      atomicAdd(grad_x + jr3 + idx110, wgt110 * g);
      atomicAdd(grad_x + jr3 + idx111, wgt111 * g);
    }
  }
}

void spherical_trilinear_devoxelize(int b, int c, int n, int r, int r2, int r3,
                          bool training, const float *coords, const float *feat,
                          const int *g_inds, int *inds, float *wgts, float *outs) {
    spherical_trilinear_devoxelize_kernel<<<b, optimal_num_threads(n)>>>(
      b, c, n, r, r2, r3, training, coords, feat, g_inds, inds, wgts, outs);
  CUDA_CHECK_ERRORS();
}

void spherical_trilinear_devoxelize_grad(int b, int c, int n, int r3, const int *inds,
                               const float *wgts, const float *grad_y,
                               float *grad_x) {
  spherical_trilinear_devoxelize_grad_kernel<<<b, optimal_num_threads(n)>>>(
      b, c, n, r3, inds, wgts, grad_y, grad_x);
  CUDA_CHECK_ERRORS();
}
