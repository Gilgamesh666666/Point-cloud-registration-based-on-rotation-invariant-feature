#include<stdio.h>
#include "../cuda_utils.cuh"
//[b,c,n], [b,c,m], float[b,k,n], int[b,k,n]
//result初始化要非常大, 默认为UNDEFINE_VALUE
__global__ void KnnKernel(int b, int c, int n, int m, int k,
	const float *__restrict__ xyz1,const float *__restrict__ xyz2,
	float *__restrict__ result,int *__restrict__ result_i){
	
	int batchsize = blockIdx.x;
	int index = threadIdx.x;
	int stride = blockDim.x;
	xyz1 += batchsize*c*n;
	xyz2 += batchsize*c*m;
	result += batchsize*k*n;
	result_i += batchsize*k*n;
	for(int i=index;i<n;i+=stride)
	{
		for(int j=0;j<m;j++)
		{
			float d = 0;
			for(int p=0;p<c;p++)
			{
				d += (xyz1[i + p*n] - xyz2[j + p*m]) * (xyz1[i + p*n] - xyz2[j + p*m]);
            }
            //insert sort
            //默认result初始化非常大
            float k_d = result[i + (k-1)*n];
            if(d < k_d)
            {
                result[i + (k-1)*n]=d;
                result_i[i + (k-1)*n] = j;
            }
            for(int q=k-1;q>0;q--)
            {
                if(result[i + q*n] < result[i + (q-1)*n])
                {
                    float temp = result[i + q*n];
                    result[i + q*n] = result[i + (q-1)*n];
                    result[i + (q-1)*n] = temp;
                
                    int temp_i = result_i[i + q*n];
                    result_i[i + q*n] = result_i[i + (q-1)*n];
                    result_i[i + (q-1)*n] = temp_i;
                }
            }
			
		}
	}
}
//[b,c,n],[b,c,m],float[b,k,n] int[b,k,n]
//无意义的值用dist=10000来默认,grad为0
__global__ void KnnGradKernel(int b,int c,int n,int m,int k,
	const float *__restrict__ xyz1,const float *__restrict__ xyz2,
	const float *__restrict__ grad_dist1,const int *__restrict__ idx1,
	float *__restrict__ grad_xyz1,float *__restrict__ grad_xyz2){
	int batchsize = blockIdx.x;
	int index = threadIdx.x;
	int stride = blockDim.x;
	xyz1 += batchsize*c*n;
	xyz2 += batchsize*c*m;
	grad_xyz1 += batchsize*c*n;
	grad_xyz2 += batchsize*c*m;
	grad_dist1 += batchsize*k*n;
	idx1 += batchsize*k*n;
	for (int i=index;i<n;i+=stride){
        for(int q=0;q<k;q++){
            float g = grad_dist1[i + q*n]*2;
            if (g>=2*10000)continue; //UNDEFINE_VALUE=10000
            int id = idx1[i + q*n];
                
            for (int p=0;p<c;p++)
            {
                atomicAdd(grad_xyz1 + i + p*n, g*(xyz1[i + p*n]-xyz2[id + p*m]));
                atomicAdd(grad_xyz2 + id + p*m, -(g*(xyz1[i + p*n]-xyz2[id + p*m])));
            }
        }
	}
}


void knn_kernel(int b, int c, int n, int m, int k, const float *xyz1_data, const float *xyz2_data,
	float *dist1_data, float *dist2_data,
	int *idx1_data, int *idx2_data){
    KnnKernel<<<b,optimal_num_threads(n)>>>(b,c,n,m,k,xyz1_data,xyz2_data,dist1_data,idx1_data);
    KnnKernel<<<b,optimal_num_threads(m)>>>(b,c,m,n,k,xyz2_data,xyz1_data,dist2_data,idx2_data);
    CUDA_CHECK_ERRORS();
}
void knn_grad_kernel(int b,int c,int n,int m,int k,
    const float *xyz1_data,const float *xyz2_data,
    float *gradxyz1_data,float *gradxyz2_data,
    const float *graddist1_data,const float *graddist2_data,
    const int *idx1_data,const int *idx2_data){

    KnnGradKernel<<<b,optimal_num_threads(n)>>>(b,c,n,m,k,xyz1_data,xyz2_data,graddist1_data,idx1_data,gradxyz1_data,gradxyz2_data);
    KnnGradKernel<<<b,optimal_num_threads(m)>>>(b,c,m,n,k,xyz2_data,xyz1_data,graddist2_data,idx2_data,gradxyz2_data,gradxyz1_data);
	
    CUDA_CHECK_ERRORS();
}