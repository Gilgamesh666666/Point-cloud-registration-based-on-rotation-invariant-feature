#ifndef _SPHERICAL_VOX_CUH
#define _SPHERICAL_VOX_CUH

// CUDA function declarations
void spherical_avg_voxelize(int b, int c, int n, int r, int r2, int r3, const float *coords,
                  const float *feat, int *ind, int *cnt, float *out);
void spherical_avg_voxelize_grad(int b, int c, int n, int s, const int *idx,
                       const int *cnt, const float *grad_y, float *grad_x);

#endif
