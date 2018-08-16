#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

//Constant Declarations:
#define POP_SIZE 8
#define CHANCE 3
#define PERCENT_CROSS 0.2

//Function declarations:
__global__ void initialization(char **population, char *target, int targetSize, char *charmap,int charmapSize, unsigned int *seed, curandState_t* states);
void fitnessCalculation();
void evolution();
void mutation(char *mutant, int n);

void printPopulation();
char randChar();
int randNumb(int n);
__device__ char randCharDev(int targetSize);
__device__ int randNumbDev(int n);

//Global Variables:
char *target = "Hello";
char *population[POP_SIZE];
char *charmap = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*(_-)+=[]{}<>|;:',./?~` ";
int fitness[POP_SIZE];
int best = 500;
int fit = 0;

int main()
{
	
	srand((unsigned int)time(NULL));
	
	//Variables

//	char *target = "Hello";					//CPU
	char *d_target;							//GPU
//	char *population[POP_SIZE];			//CPU
	char *d_population[POP_SIZE];			//GPU

//	char *charmap = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*(_-)+=[]{}<>|;:',./?~` ";	//CPU
	char *d_charmap = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*(_-)+=[]{}<>|;:',./?~` ";//GPU

//	int fitness[POP_SIZE];	//CPU
//	int d_fitness[POP_SIZE];//GPU
//	int best = 500;			//CPU
//	int d_best = 500;			//GPU
//	int fit = 0;				//CPU
//	int d_fit = 0;				//GPU

	//CPU memory allocation

	for(int i = 0; i < POP_SIZE; i++)	//CPU
	{
		population[i] =(char*)malloc(sizeof(char)*strlen(target));
	}
	for(int j = 0; j < POP_SIZE; j++)	//GPU
	{
		d_population[j] =(char*)malloc(sizeof(char)*strlen(target));
	}


	//GPU memory allocation 
	cudaMalloc((char **)&d_target,sizeof(char)*strlen(target));
	cudaMalloc((void **)&d_population,sizeof(population));
	for (int k = 0; k < POP_SIZE; k++)
	{
		cudaMalloc((char**)&d_population[k],sizeof(char)*strlen(target));
	}
//	cudaMalloc((void**)&d_best,sizeof(int));
//	cudaMalloc((void**)&d_fit,sizeof(int));
	cudaMalloc((char**)&d_charmap,sizeof(char)*strlen(charmap));
	
	//Sending data to GPU
	cudaMemcpy(d_target,target,strlen(target)*sizeof(char),cudaMemcpyHostToDevice);
	cudaMemcpy(d_charmap,charmap,strlen(charmap)*sizeof(char),cudaMemcpyHostToDevice);
//	cudaMemcpy(d_best,best,sizeof(int),cudaMemcpyHostToDevice);
//	cudaMemcpy(d_fit,fit,sizeof(int),cudaMemcpyHostToDevice);


	//Initializing random seed and allocating it both on CPU and GPU:
	  curandState_t* states;
	  unsigned int *h_seed = (unsigned int*)malloc(sizeof(unsigned int)*POP_SIZE);
	  srand(time(NULL));
	  for(int i=0;i<POP_SIZE;i++)
	  {
	    h_seed[i] = rand()%100000;
	  }
	
	  cudaMalloc((void**) &states, POP_SIZE * sizeof(curandState_t));
	  unsigned int *d_seed;
	
	  cudaMalloc((void**)&d_seed, sizeof(unsigned int)*POP_SIZE);
	  cudaMemcpy(d_seed, h_seed, sizeof(unsigned int)*POP_SIZE,cudaMemcpyHostToDevice);

	//Initializing population:
	clock_t start_pop = clock();
	initialization<<<1,POP_SIZE>>>(population,target,strlen(target),d_charmap,strlen(charmap),d_seed,states);
	clock_t finished_pop = clock();
	double popInit_time = ((double)(finished_pop - start_pop)/CLOCKS_PER_SEC);

	// Cleaning up random init:
	  cudaFree(states); cudaFree(d_seed);
	  free(h_seed);


	//Copy result back:
	cudaMemcpy(population,d_population,sizeof(population),cudaMemcpyDeviceToHost);

	//Clean GPU:
	cudaFree(d_target); cudaFree(d_charmap);
	for(int p = 0; p < POP_SIZE; p++)
      cudaFree(d_population[p]);
    cudaFree(d_population);


	clock_t start_fitCalc = clock();
	fitnessCalculation();
	clock_t finished_fitCalc = clock();
	double fitCalc_time = ((double)(finished_fitCalc - start_fitCalc)/CLOCKS_PER_SEC);

	clock_t start_evol = clock();
	evolution();
	clock_t finished_evol = clock();
	double evol_time = ((double)(finished_evol - start_evol)/CLOCKS_PER_SEC);

	printPopulation();

	while(best)
	{
		evolution();
		fitnessCalculation();
		printPopulation();
	}

	printf("InitTime: %f FitTime: %f Evol: %f\n",popInit_time,fitCalc_time,evol_time);

	return 0;
}

//	================================================ GA Functions ===============================================	//
// CUDA initialization:
__global__ void initialization(char **population, char *target, int targetSize, char *charmap,int charmapSize, unsigned int *seed, curandState_t* states)
{
	curand_init(seed[threadIdx.x],threadIdx.x,0,&states[threadIdx.x]);
	int randNumb = curand(&states[threadIdx.x])% charmapSize;

	int index = blockDim.x * blockIdx.x + threadIdx.x;
		for(int j = 0; j < targetSize; j++)
		{
			population[index][j] = charmap[randNumb];
			randNumb = curand(&states[(threadIdx.x)+randNumb])% charmapSize;
		}
	population[index][targetSize] = '\0';

}

//The lesser the better. 0 is the optimal value:
void fitnessCalculation()
{
	int i = POP_SIZE-1;
	int j;
	while(i >= 0)
	{
		fitness[i] = 0;
		for (j = (int)sizeof(target)-1; j >= 0; j--) {
			fitness[i] += abs(target[j]-population[i][j]);
		}
		if(fitness[i] < best)
		{
			best = fitness[i];
			fit = i;
		}
		i--;
	}
}

void evolution()
{

	int j = 0;
	char *newBorn = population[fit];
	int lucky;
	for(int i = 0; i < POP_SIZE; i++)
	{
		while(1)
		{
			if(j >= POP_SIZE)
			{
				j = 0;
			}
			//Selection:
			if(fitness[j] <= best)
			{
				newBorn = population[j];
				
				//Mutation:
				lucky = randNumb(POP_SIZE);
				if(lucky != j)
				{
					mutation(population[lucky],randNumb(CHANCE));
				}


				fitnessCalculation();
				j++;
				break;
			}
			//Crossover:
			else
			{
				for(int n = 0; n < strlen(target)*PERCENT_CROSS; n++)
				{
					population[j][randNumb(strlen(target))] = newBorn[randNumb(strlen(target))];
				}
			}
			j++;
		}
	}
}

void mutation(char *mutant, int n)
{
	for(int k = 0; k < n; k++)
	{
		mutant[randNumb(strlen(target))] = randChar();
	}
}

//	================================================ End GA Steps ===============================================	//


//	Helper Functions:

void printPopulation()
{
	for(int i = 0; i < POP_SIZE; i++)
	{
		printf("P: %s F: %d\n",population[i],fitness[i]);
	}
	printf("Target: %s\n",target);
	printf("Best: %s - %d\n",population[fit],best);
}

char randChar()
{

	return charmap[randNumb(strlen(charmap))];
}

int randNumb(int n)
{
	return (rand()%(int)(n));
}

