#ifndef _PPF_CUH
#define _PPF_CUH

// CUDA function declarations
void spherical_ppf(int b, int n, const float *coords,const float *center,
                  const float *normals, const float *center_normal,float *feat);

#endif
