#include <stdio.h>
#include <stdlib.h>

#include "../cuda_utils.cuh"

/*
  Function: voxelize ppf (just forward)
  Args:
    b   : batch size
    n   : number of points
    r   : voxel resolution
    coords : coords of each point, FloatTensor[b, 3, n] 原始coords
    center : center point, FloatTensor[b, 3, n]
    normals : normals of each point, FloatTensor[b, 3, n] 原始coords
    center_normal : normal of center point, FloatTensor[b, 3, n]
  output:
    feat: features, FloatTensor[b, 4, n]
*/
__global__ void spherical_ppf_kernel(int b, int n, 
                             const float *__restrict__ coords,
                             const float *__restrict__ center,
                             const float *__restrict__ normals,
                             const float *__restrict__ center_normal,
                             float *__restrict__ feat){
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index*3*n;
  center += batch_index*3*n;
  normals += batch_index*3*n;
  center_normal += batch_index*3*n;
  feat += batch_index*4*n;
  
  for(int i = index;i < n;i += stride)
  {
    //printf("(%d)", i);
    float x = coords[i];
    //printf("(%.2f)", x);
    float y = coords[i + n];
    float z = coords[i + n + n];
    float nx = normals[i];
    float ny = normals[i + n];
    float nz = normals[i + n + n];

    float cx = center[i];
    float cy = center[i + n];
    float cz = center[i + n + n];
    
    float cnx = center_normal[i];
    float cny = center_normal[i + n];
    float cnz = center_normal[i + n + n];

    float dx = cx - x;
    float dy = cy - y;
    float dz = cz - z;
    float d_norm = max(sqrt(dx*dx+dy*dy+dz*dz), pow(10, -20));
    dx /= d_norm;
    dy /= d_norm;
    dz /= d_norm;

    float n1_norm = sqrt(cnx*cnx+cny*cny+cnz*cnz);
    float n2_norm = sqrt(nx*nx+ny*ny+nz*nz);
    if(n2_norm <= pow(10, -10)||n1_norm <= pow(10, -10))
    {
      // undefined point
      feat[i] = 0;
      feat[i + n] = 0;
      feat[i + n + n] = 0;
      feat[i + n + n + n] = 0;
      continue;
    }
    else
    {
      cnx /=n1_norm;
      cny /=n1_norm;
      cnz /=n1_norm;

      nx /=n2_norm;
      ny /=n2_norm;
      nz /=n2_norm;
      float apha1 = acos(max(min((dx*cnx+dy*cny+dz*cnz), 1.0), -1.0));
      float apha2 = acos(max(min((dx*nx+dy*ny+dz*nz), 1.0), -1.0));
      float apha3 = acos(max(min((cnx*nx+cny*ny+cnz*nz), 1.0), -1.0));

      feat[i] = apha1;
      feat[i + n] = apha2;
      feat[i + n + n] = apha3;
      feat[i + n + n + n] = d_norm;
      // output:ppfs (b,m,RuleNumInPatch,4)
    }
  }
}

void spherical_ppf(int b, int n, const float *coords,const float *center,
                  const float *normals, const float *center_normal,float *feat) {
  spherical_ppf_kernel<<<b, optimal_num_threads(n)>>>(b, n, coords, center,
                                                      normals, center_normal, feat);
  CUDA_CHECK_ERRORS();
}
