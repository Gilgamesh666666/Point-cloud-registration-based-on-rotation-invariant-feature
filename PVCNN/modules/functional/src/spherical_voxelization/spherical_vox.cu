#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../cuda_utils.cuh"
#define PI acos(-1.0) 
/*
  Function: get how many points in each voxel grid
  Args:
    b      : batch size
    n      : number of points
    r      : voxel resolution
    r2     : = r * r
    r3     : s, voxel cube size = r ** 3
    coords : coords of each point, FloatTensor[b, 3, n] 原始coords
    ind    : voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
    在外面做norm好了,这里就当做centriod在原点,且离原点最远的点已经norm为1
*/
__global__ void spherical_grid_stats_kernel(int b, int n, int r, int r2, int r3,
                                  const float *__restrict__ coords,
                                  int *__restrict__ ind, int *cnt) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index * n * 3;
  ind += batch_index * n;
  cnt += batch_index * r3;
  //printf("r2=%d\n",r2);
  //float interval = max_norm/r = 1/r;
  for (int i = index; i < n; i += stride) {
    // if (ind[i] == -1)
    //   continue;
    // gama, alpha, beta
    float x = coords[i];
    float y = coords[i + n];
    float z = coords[i + n + n];
    float gama = sqrt(x * x + y * y + z * z);
    float alpha, beta;
    //printf("%d, %.2f, %.2f, %.2f, %.2f\n", i, x, y, z, gama);
    if((gama==0)||(gama>=1)||((z / gama) > 1)||((z / gama) < -1))
    {
      ind[i] = -1;
    }
    else
    {
      beta = acos(z / gama);
      if(beta>=PI)
      {
        ind[i]=-1;
        continue;
      }
      if(x==0&&y!=0) alpha = (y/abs(y))*PI*0.5;
      else if(x==0&&y==0) alpha = 0;
      else alpha = atan(y/x) + PI*(1-(x/abs(x)))/2;
      alpha += PI/r;
      if(alpha<0)alpha += 2*PI;
      //alpha = fmod(alpha, (2*PI));
      // voxel化
      int grid_x = floor(gama*r); //gama/interval = gama*r
      int grid_y = floor(alpha*r/2/PI);
      int grid_z = floor(beta*r/PI);
      if(grid_x>=r)grid_x = r-1;
      if(grid_y>=r)grid_y = r-1;
      if(grid_z>=r)grid_z = r-1;
      ind[i] = grid_x * r2 + grid_y * r + grid_z;
      //printf(" ind: %d ",ind[i]);
      //printf("i=%d,r2=%d,r=%d\n",i,r2,r);
      //printf("i=%d,r2=%d,r=%d,indi=%d,grid_x=%d,grid_y=%d,grid_z=%d\n", i,r2,r,tmp,grid_x,grid_y,grid_z);
      if(grid_x>=r||grid_y>=r||grid_z>=r)
      {
        printf("x=%f,y=%f,z=%f,gama=%f,alpha=%f,beta=%f,grid_x=%d,grid_y=%d,grid_z=%d, atan=%f\n",x,y,z,gama,alpha,beta,grid_x,grid_y,grid_z,atan(y/x));
      }//printf("here0");
      atomicAdd(cnt + ind[i], 1);
    }
    //printf("%d, %d\n", i, ind[i]);
  }
}

/*
  Function: average pool voxelization (forward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    s   : voxel cube size = voxel resolution ** 3
    ind : voxel index of each point, IntTensor[b, n]
    cnt : #points in each voxel index, IntTensor[b, s]
    feat: features, FloatTensor[b, c, n]
    out : outputs, FloatTensor[b, c, s]
*/
__global__ void spherical_avg_voxelize_kernel(int b, int c, int n, int s,
                                    const int *__restrict__ ind,
                                    const int *__restrict__ cnt,
                                    const float *__restrict__ feat,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  feat += batch_index * c * n;
  out += batch_index * c * s;
  cnt += batch_index * s;
  for (int i = index; i < n; i += stride) {
    //printf("here");
    int pos = ind[i];
    if (pos == -1)
      continue;
    int cur_cnt = cnt[pos];
    //__syncthreads();
    //printf(" (%d) ",(s-pos));
    if (cur_cnt > 0) {
      float div_cur_cnt = 1.0 / static_cast<float>(cur_cnt);
      //printf(" (%d, %d) ",i, n);
      for (int j = 0; j < c; j++) {
        //__syncthreads();
        atomicAdd(out + j * s + pos, feat[j * n + i]*div_cur_cnt);
        //atomicAdd(out + j * s + pos, 0);
        //out = out + j * s + pos;
        //printf(" (%d, %d) ",j, c);
      }
    }
    //__syncthreads();
  }
  //printf("there");
}

/*
  Function: average pool voxelization (backward)
  Args:
    b      : batch size
    c      : #channels
    n      : number of points
    r3     : voxel cube size = voxel resolution ** 3
    ind    : voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
    grad_y : grad outputs, FloatTensor[b, c, s]
    grad_x : grad inputs, FloatTensor[b, c, n]
*/
__global__ void spherical_avg_voxelize_grad_kernel(int b, int c, int n, int r3,
                                         const int *__restrict__ ind,
                                         const int *__restrict__ cnt,
                                         const float *__restrict__ grad_y,
                                         float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  grad_x += batch_index * c * n;
  grad_y += batch_index * c * r3;
  cnt += batch_index * r3;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i];
    if (pos == -1)
      continue;
    int cur_cnt = cnt[pos];
    if (cur_cnt > 0) {
      float div_cur_cnt = 1.0 / static_cast<float>(cur_cnt);
      for (int j = 0; j < c; j++) {
        atomicAdd(grad_x + j * n + i, grad_y[j * r3 + pos] * div_cur_cnt);
      }
    }
  }
}

void spherical_avg_voxelize(int b, int c, int n, int r, int r2, int r3, const float *coords,
                            const float *feat, int *ind, int *cnt, float *out) {
  spherical_grid_stats_kernel<<<b, optimal_num_threads(n)>>>(b, n, r, r2, r3, coords, ind,
                                                   cnt);
  //cudaDeviceSynchronize();
  CUDA_CHECK_ERRORS();
  //printf("%s\n",cudaGetErrorString(cudaGetLastError()));
  spherical_avg_voxelize_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, r3, ind, cnt,
                                                     feat, out);
  //cudaDeviceSynchronize();
  // printf("%s\n",cudaGetErrorString(cudaGetLastError()));
  CUDA_CHECK_ERRORS();
}

void spherical_avg_voxelize_grad(int b, int c, int n, int s, const int *ind,
                       const int *cnt, const float *grad_y, float *grad_x) {
  spherical_avg_voxelize_grad_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, s, ind, cnt,
                                                          grad_y, grad_x);
  CUDA_CHECK_ERRORS();
}
