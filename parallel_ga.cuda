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
void initialization();
void fitnessCalculation();
void evolution();
void mutation(char *mutant, int n);

void printPopulation();
char randChar();
int randNumb(int n);

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
	int d_fitness[POP_SIZE];//GPU
//	int best = 500;			//CPU
	int d_best = 500;			//GPU
//	int fit = 0;				//CPU
	int d_fit = 0;				//GPU

	//CPU memory allocation

	for(int i = 0; i < POP_SIZE; i++)	//CPU
	{
		population[i] = malloc(sizeof(char)*strlen(target));
	}
	for(int j = 0; j < POP_SIZE; j++)	//GPU
	{
		d_population[j] = malloc(sizeof(char)*strlen(target));
	}


	//GPU memory allocation 
	cudaMalloc((void **)&d_target,sizeof(char)*strlen(target));
	cudaMalloc((void **)&d_population,sizeof(population));
	for (int k = 0; k < POP_SIZE; k++)
	{
		cudaMalloc((void**)&d_population[k],sizeof(char)*strlen(target));
	}
//	cudaMalloc((void**)&d_best,sizeof(int));
//	cudaMalloc((void**)&d_fit,sizeof(int));
	cudaMalloc((void**)&d_charmap,sizeof(char)*strlen(charmap));
	
	//Sending data to GPU
	cudaMemcpy(d_target,target,strlen(target)*sizeof(char),cudaMemcpyHostToDevice);
	cudaMemcpy(d_charmap,charmap,strlen(charmap)*sizeof(char),cudaMemcpyHostToDevice);
//	cudaMemcpy(d_best,best,sizeof(int),cudaMemcpyHostToDevice);
//	cudaMemcpy(d_fit,fit,sizeof(int),cudaMemcpyHostToDevice);

	//Initializing population:
	initialization<<<1,POP_SIZE>>>(population,target);

	//Copy result back:
	cudaMemcpy(population,d_population,sizeof(population),cudaMemcpyDeviceToHost);

	//Clean GPU:
	cudaFree(d_target); cudaFree(d_charmap);
	for(int p = 0; p < POP_SIZE; p++)
      cudaFree(d_population[p]);
    cudaFree(d_population);

	fitnessCalculation();
	printPopulation();

	while(best)
	{
		evolution();
		fitnessCalculation();
		printPopulation();
	}

	return 0;
}

//	================================================ GA Functions ===============================================	//
// CUDA initialization:
__global__ void initialization(char **population, char *target)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
		for(int j = 0; j < strlen(target); j++)
		{
			population[index][j] = randChar();
		}
		population[index][strlen(target)] = '\0';
	}
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

	return charmap[randNumb(strlen(charmap)-1)];
}

int randNumb(int n)
{
	return (rand()%(int)(n));
}

