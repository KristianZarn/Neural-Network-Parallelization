#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "neuralnetwork_pthreads.h"
#include "helpers.h"

#define HAVE_STRUCT_TIMESPEC
#include <pthread.h>
#define errorexit(errcode, errstr) \
	fprintf(stderr , "%s: %d\n", errstr,errcode); \
	exit(1);

typedef struct updateParamData {
	int  threadId;
	int size;
	double * param;
	double * grad;
	double alpha;
} updateParamData;

typedef struct costFunctionData {
	int rank;
	int T1rows;
	int T1cols;
	int numExamples;
	int T2rows;
	int T2cols;
	double ** T1;
	double ** T2;
	double ** T1grad;
	double ** T2grad;
	double ** Y;
	double ** X;
	double cost;
} costFunctionData;

void sigmoid(double* vector, int len) {
	/*
	Na elementih vektorja izracuna logisticno funkcijo in rezultate zapise v isti vektor
	*/

	for (int i = 0; i < len; i++) {
		vector[i] = 1.0 / (1.0 + exp(vector[i] * (-1)));
	}
}

void sigmoidGradient(double* vector, int len) {
	/*
	Na elementih izracuna odvod logisticne funckije in rezultate zapise v isti vektor
	*/

	sigmoid(vector, len);
	for (int i = 0; i < len; i++) {
		vector[i] = vector[i] * (1.0 - vector[i]);
	}
}

double * predict(double ** X, int Xrows, int Xcols, double ** Theta1, int T1rows, int T1cols, double ** Theta2, int T2rows, int T2cols) {
	/*
	Predikcija oznak vhodnih primerov z podanimi parametri
	*/

	double * result = (double*)calloc(Xrows, sizeof(double));

	for (int i = 0; i < Xrows; i++) { // Gremo v zanki cez vsak testni primer
		double * a2 = (double*)calloc(T1rows, sizeof(double));
		double * a3 = (double*)calloc(T2rows, sizeof(double));

		// aktivacija drugega nivoja a2
		for (int theta_i = 0; theta_i < T1rows; theta_i++) {
			a2[theta_i] = Theta1[theta_i][0]; //v bistvu moram vektorju xi dodati enico na zacetek tako da je prvi produkt enak prvi theti v vrstici.
			for (int theta_j = 1; theta_j < T1cols; theta_j++) {
				a2[theta_i] += (Theta1[theta_i][theta_j] * X[i][theta_j - 1]);
			}
		}
		sigmoid(a2, T1rows);

		// aktivacija tretjega nivoja a3
		for (int theta_i = 0; theta_i < T2rows; theta_i++) {
			a3[theta_i] = Theta2[theta_i][0]; //v bistvu moram vektorju xi dodati enico na zacetek tako da je prvi produkt enak prvi theti v vrstici.
			for (int theta_j = 1; theta_j < T2cols; theta_j++) {
				a3[theta_i] += (Theta2[theta_i][theta_j] * a2[theta_j - 1]);
			}
		}
		sigmoid(a3, T2rows);

		// Index najvecjega izhodnega nevrona predstavlja ustrezno crko
		double max = a3[0];
		int indeks = 0;
		for (int count = 1; count < T2rows; count++) {
			if (a3[count] > max) {
				max = a3[count];
				indeks = count;
			}
		}
		result[i] = indeks; // shranim rezultat klasifikacije

		free(a2);
		free(a3);
	}

	return result;
}

void randInitializeWeights(double * param, int paramSize) {
	/*
	Nakljucna inicializacija parametrov
	*/

	double epsilon_init = 0.12;
	for (int i = 0; i < paramSize; i++) {
		double r = (double)rand() / (double)RAND_MAX;
		param[i] = (r * 2.0 * epsilon_init) - epsilon_init;
	}
}

void debugInitializeWeights(double * param, int paramSize) {
	/*
	Inicializacija parametrov za debagiranje
	*/

	for (int i = 0; i < paramSize; i++) {
		param[i] = sin((double)i);
	}
}

void * costFunctionParallel(void * arg) {
	int rank = ((costFunctionData*)arg)->rank;
	int T1rows = ((costFunctionData*)arg)->T1rows;
	int T1cols = ((costFunctionData*)arg)->T1cols;
	int numExamples = ((costFunctionData*)arg)->numExamples;
	int T2rows = ((costFunctionData*)arg)->T2rows;
	int T2cols = ((costFunctionData*)arg)->T2cols;
	double ** T1 = ((costFunctionData*)arg)->T1;
	double ** T2 = ((costFunctionData*)arg)->T2;
	double ** Y = ((costFunctionData*)arg)->Y;
	double ** X = ((costFunctionData*)arg)->X;

	double ** T1grad = allocateMatrix(T1rows, T1cols);
	double ** T2grad = allocateMatrix(T2rows, T2cols);

	((costFunctionData*)arg)->T1grad = T1grad;
	((costFunctionData*)arg)->T2grad = T2grad;

	int m = (rank*numExamples) / NUM_THREADS;
	int M = ((rank + 1)*numExamples) / NUM_THREADS - 1;


	for (int t = m; t <= M; t++) {
		double * z2 = (double*)calloc(T1rows, sizeof(double));
		double * a2 = (double*)calloc(T1rows, sizeof(double));
		double * a3 = (double*)calloc(T2rows, sizeof(double));

		// feedforward
		for (int theta_i = 0; theta_i < T1rows; theta_i++) {
			z2[theta_i] = T1[theta_i][0];
			for (int theta_j = 1; theta_j < (T1cols); theta_j++) {
				z2[theta_i] += (T1[theta_i][theta_j] * X[t][theta_j - 1]);
			}
		}
		memcpy(a2, z2, T1rows * sizeof(double));
		sigmoid(a2, T1rows);

		for (int theta_i = 0; theta_i < T2rows; theta_i++) {
			a3[theta_i] = T2[theta_i][0];
			for (int theta_j = 1; theta_j < (T2cols); theta_j++) {
				a3[theta_i] += (T2[theta_i][theta_j] * a2[theta_j - 1]);
			}
		}
		sigmoid(a3, T2rows);

		// pristej k cost function
		for (int i = 0; i < T2rows; i++) {
			((costFunctionData*)arg)->cost += ((-1.0) * Y[t][i] * log(a3[i])) - ((1.0 - Y[t][i]) * log(1.0 - a3[i]));
		}

		// backpropagation
		double * d3 = (double*)calloc(T2rows, sizeof(double));
		double * d2 = (double*)calloc(T1rows, sizeof(double));

		for (int i = 0; i < T2rows; i++) {
			d3[i] = a3[i] - Y[t][i];
		}

		sigmoidGradient(z2, T1rows);
		for (int i = 0; i < T1rows; i++) {
			for (int j = 0; j < T2rows; j++) {
				d2[i] += T2[j][i + 1] * d3[j];
			}
			d2[i] = d2[i] * z2[i];
		}

		// pristej gradientu
		for (int i = 0; i < T1rows; i++) {
			T1grad[i][0] += d2[i];
			for (int j = 1; j < T1cols; j++) {
				T1grad[i][j] += d2[i] * X[t][j - 1];
			}
		}

		for (int i = 0; i < T2rows; i++) {
			T2grad[i][0] += d3[i];
			for (int j = 1; j < T2cols; j++) {
				T2grad[i][j] += d3[i] * a2[j - 1];
			}
		}

		free(z2);
		free(a2);
		free(a3);
		free(d3);
		free(d2);
	}

	return NULL;
}

double costFunction(double * grad, double * param, int paramSize, double ** X, int Xrows, int Xcols, int hiddenLayerSize, double ** Y, int yLabels, double lambda) {
	/*
	Poracuna cost function in gradient parametrov, ki je uporabljen v optimizaciji
	*/

	// reshape parameters
	int iparam = 0;
	int T1rows = hiddenLayerSize;
	int T1cols = Xcols + 1;
	int numExamples = Xrows;
	double ** T1 = allocateMatrix(T1rows, T1cols);
	for (int j = 0; j < T1cols; j++) {
		for (int i = 0; i < T1rows; i++) {
			T1[i][j] = param[iparam];
			iparam++;
		}
	}
	int T2rows = yLabels;
	int T2cols = hiddenLayerSize + 1;
	double ** T2 = allocateMatrix(T2rows, T2cols);
	for (int j = 0; j < T2cols; j++) {
		for (int i = 0; i < T2rows; i++) {
			T2[i][j] = param[iparam];
			iparam++;
		}
	}

	// akumulatorji za gradient
	double ** T1grad = allocateMatrix(T1rows, T1cols);
	double ** T2grad = allocateMatrix(T2rows, T2cols);


	// vsaka nit dobi svoj delez ucne mnozice
	double cost = 0;

	pthread_t threads[NUM_THREADS];
	costFunctionData p[NUM_THREADS];

	for (int i = 0; i < NUM_THREADS; i++) {
		p[i].rank = i;
		p[i].T1rows = T1rows;
		p[i].T1cols = T1cols;
		p[i].numExamples = numExamples;
		p[i].T2rows = T2rows;
		p[i].T2cols = T2cols;
		p[i].T1 = T1;
		p[i].T2 = T2;
		p[i].Y = Y;
		p[i].X = X;
		p[i].cost = 0;
		pthread_create(&threads[i], NULL, costFunctionParallel, (void*)&p[i]);
	}

	// glavna nit zdruzi poracunane gradiente
	for (int id = 0; id < NUM_THREADS; id++) {
		pthread_join(threads[id], NULL);
		for (int i = 0; i < T1rows; i++) {
			for (int j = 0; j < T1cols; j++) {
				T1grad[i][j] += p[id].T1grad[i][j];
			}
		}
		for (int i = 0; i < T2rows; i++) {
			for (int j = 0; j < T2cols; j++) {
				T2grad[i][j] += p[id].T2grad[i][j];
			}
		}
		cost += p[id].cost;

		freeMatrix(p[id].T1grad, T1rows);
		freeMatrix(p[id].T2grad, T2rows);
	}


	// normalizacija cost funkcije
	cost = (1.0 / (double)numExamples) * cost;

	// regularizacija cost funkcije
	double costReg = 0.0;
	for (int i = 0; i < T1rows; i++) {
		for (int j = 1; j < T1cols; j++) {
			costReg += T1[i][j] * T1[i][j];
		}
	}
	for (int i = 0; i < T2rows; i++) {
		for (int j = 1; j < T2cols; j++) {
			costReg += T2[i][j] * T2[i][j];
		}
	}
	cost += (lambda / (2 * (double)numExamples)) * costReg;

	// normalizacija gradienta
	for (int i = 0; i < T1rows; i++) {
		for (int j = 0; j < T1cols; j++) {
			T1grad[i][j] = T1grad[i][j] / (double)numExamples;
		}
	}
	for (int i = 0; i < T2rows; i++) {
		for (int j = 0; j < T2cols; j++) {
			T2grad[i][j] = T2grad[i][j] / (double)numExamples;
		}
	}

	// regularizacija gradienta
	for (int i = 0; i < T1rows; i++) {
		for (int j = 1; j < T1cols; j++) {
			T1grad[i][j] += (lambda / (double)numExamples) * T1[i][j];
		}
	}
	for (int i = 0; i < T2rows; i++) {
		for (int j = 1; j < T2cols; j++) {
			T2grad[i][j] += (lambda / (double)numExamples) * T2[i][j];
		}
	}

	// unroll gradientov
	int igrad = 0;
	for (int j = 0; j < T1cols; j++) {
		for (int i = 0; i < T1rows; i++) {
			grad[igrad] = T1grad[i][j];
			igrad++;
		}
	}
	for (int j = 0; j < T2cols; j++) {
		for (int i = 0; i < T2rows; i++) {
			grad[igrad] = T2grad[i][j];
			igrad++;
		}
	}

	freeMatrix(T1, T1rows);
	freeMatrix(T2, T2rows);
	freeMatrix(T1grad, T1rows);
	freeMatrix(T2grad, T2rows);
	return cost;
}

void * updateParam(void * args) {
	updateParamData * data = (updateParamData *)args;

	int tid = data->threadId;
	int start = (int)(data->size / (double)NUM_THREADS * tid);
	int stop = (int)(data-> size / (double)NUM_THREADS * (tid + 1));

	double * param = data->param;
	double * grad = data->grad;
	double alpha = data->alpha;

	for (int i = start; i < stop; i++)
	{
		param[i] = param[i] - alpha * grad[i];
	}

	return NULL;
}

double gradientDescent(double * param, int paramSize, int iterations, double ** X, int Xrows, int Xcols, int hiddenLayerSize, double * y, int yLabels, double lambda) {
	/*
	Optimizacija parametrov z gradientnim spustom
	*/

	// pretvori oznake y v vektorje
	double ** Y = allocateMatrix(Xrows, yLabels);
	for (int i = 0; i < Xrows; i++) {
		int label = (int)y[i];
		Y[i][label] = 1.0;
	}

	// optimizacija
	double cost = 0;
	double * grad = (double*)calloc(paramSize, sizeof(double));

	for (int iter = 0; iter < iterations; iter++) {
		cost = costFunction(grad, param, paramSize, X, Xrows, Xcols, hiddenLayerSize, Y, yLabels, lambda);

		// double alpha = 1.0 - ((double)iter / (double)iterations)*0.3;
		double alpha = 1.0;

		pthread_t threads[NUM_THREADS];
		updateParamData threadArgs[NUM_THREADS];
		for (int i = 0; i < NUM_THREADS; i++) {
			threadArgs[i].threadId = i;
			threadArgs[i].size = paramSize;
			threadArgs[i].param = param;
			threadArgs[i].grad = grad;
			threadArgs[i].alpha = alpha;
			pthread_create(&threads[i], NULL, updateParam, (void *) &threadArgs[i]);
		}

		for (int i = 0; i < NUM_THREADS; i++) {
			pthread_join(threads[i], NULL);
		}

		printf("Iteration: %4d | Alpha:  %.4f | Cost: %4.6f\n", iter + 1, alpha, cost);
		fflush(stdout);
	}

	free(grad);
	freeMatrix(Y, Xrows);
	return cost;
}
