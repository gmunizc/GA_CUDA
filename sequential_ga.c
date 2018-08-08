#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

//Constant Declarations:
#define POP_SIZE 8
#define CHANCE 3
#define PERCENT_CROSS 0.2

//Function declarations:
void initialization();
void fitnessCalculation();
void evolution();
void mutation(char *mutant, int n);

int isDone();
void printPopulation();
char randChar();
int randNumb(int n);

//Global Variables:
char *target = "Hello!";
char *population[POP_SIZE];
char *charmap = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*(_-)+=[]{}<>|;:',./?~` ";
int fitness[POP_SIZE];
int best = 500;
int fit = 0;

int main()
{
	//Initializing population:
	srand((unsigned int)time(NULL));
	initialization();

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

void initialization()
{
	int i = 0;
	while(i < POP_SIZE)
	{
		population[i] = malloc(sizeof(target));
		for(int j = 0; j < strlen(target); j++)
		{
			population[i][j] = randChar();
		}
		population[i][strlen(target)] = '\0';
		i++;
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

