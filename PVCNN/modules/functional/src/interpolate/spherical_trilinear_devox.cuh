#ifndef _SPHERICAL_TRILINEAR_DEVOX_CUH
#define _SPHERICAL_TRILINEAR_DEVOX_CUH

// CUDA function declarations
void spherical_trilinear_devoxelize(int b, int c, int n, int r, int r2, int r3,
                          bool is_training, const float *coords,
                          const float *feat, const int *__restrict__ g_inds, int *inds, float *wgts,
                          float *outs);
void spherical_trilinear_devoxelize_grad(int b, int c, int n, int r3, const int *inds,
                               const float *wgts, const float *grad_y,
                               float *grad_x);

#endif
