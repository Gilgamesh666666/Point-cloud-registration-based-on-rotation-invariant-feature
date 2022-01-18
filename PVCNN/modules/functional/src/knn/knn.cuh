#ifndef _KNN_CUH
#define _KNN_CUH
#define UNDEFINE_VALUE 10000;
void knn_kernel(int b, int c, int n, int m, int k, const float *xyz1_data, const float *xyz2_data,
	float *dist1_data, float *dist2_data,
	int *idx1_data, int *idx2_data);
void knn_grad_kernel(int b,int c,int n,int m,int k,
    const float *xyz1_data,const float *xyz2_data,
    float *gradxyz1_data,float *gradxyz2_data,
    const float *graddist1_data,const float *graddist2_data,
    const int *idx1_data,const int *idx2_data);

#endif