#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>

__global__ void randWork(unsigned int *seed, curandState_t* states, int *d_rnumbs)
{
   curand_init(seed[threadIdx.x],threadIdx.x,0,&states[threadIdx.x]);
   d_rnumbs[threadIdx.x] = curand(&states[threadIdx.x])% 100;

}

int main(){

  int nThreads = 10;

  curandState_t* states;
  unsigned int *h_seed = (unsigned int*)malloc(sizeof(unsigned int)*nThreads);
  srand(time(NULL));
  for(int i=0;i<nThreads;i++)
  {
    h_seed[i] = rand()%100000;
  }

  int *rnumbs = (int*)malloc(sizeof(int)*nThreads);
  int *d_rnumbs = (int*)malloc(sizeof(int)*nThreads);
  cudaMalloc((void**)&d_rnumbs, sizeof(int)*nThreads);

  cudaMalloc((void**) &states, nThreads * sizeof(curandState_t));
  unsigned int *d_seed;

  cudaMalloc((void**)&d_seed, sizeof(unsigned int)*nThreads);
  cudaMemcpy(d_seed, h_seed, sizeof(unsigned int)*nThreads,cudaMemcpyHostToDevice);

  // veja somente parametros d_seed e states
  randWork<<<1,nThreads>>>(d_seed ,states,d_rnumbs);

  cudaMemcpy(rnumbs, d_rnumbs, sizeof(int)*nThreads,cudaMemcpyDeviceToHost);

  cudaFree(states); cudaFree(d_seed);
  free(h_seed);


  printf("Random Numbers:\n");
  for (int i = 0; i < nThreads; i++)
    printf("%d: %d\n", i, rnumbs[i]);

  return 0;
}
