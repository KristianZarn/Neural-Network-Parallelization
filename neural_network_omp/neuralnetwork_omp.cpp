#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include "neuralnetwork_omp.h"
#include "helpers.h"

void sigmoid(double* vector, int len) {
	/*
	Na elementih vektorja izracuna logisticno funkcijo in rezultate zapise v isti vektor	*/

	#pragma ivdep
	for (int i = 0; i < len; i++) {
		vector[i] = 1.0 / (1.0 + exp(vector[i] * (-1)));
	}
}

void sigmoidGradient(double* vector, int len) {
	/*
	Na elementih izracuna odvod logisticne funckije in rezultate zapise v isti vektor
	*/

	sigmoid(vector, len);
	#pragma ivdep
	for (int i = 0; i < len; i++) {
		vector[i] = vector[i] * (1.0 - vector[i]);
	}
}

void sigmoid2(double** matrix, int len, int column) {
	/*
	Na elementih vektorja izracuna logisticno funkcijo in rezultate zapise v isti vektor
	*/
	#pragma ivdep
	for (int i = 0; i < len; i++) {
		matrix[i][column] = 1.0 / (1.0 + exp(matrix[i][column] * (-1)));
	}
}

void sigmoidGradient2(double** matrix, int len, int column) {
	/*
	Na elementih izracuna odvod logisticne funckije in rezultate zapise v isti vektor
	*/
	sigmoid2(matrix, len, column);
	#pragma ivdep
	for (int i = 0; i < len; i++) {
		matrix[i][column] = matrix[i][column] * (1.0 - matrix[i][column]);
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
			for (int theta_j = 1; theta_j < (T1cols + 1); theta_j++) {
				a2[theta_i] += (Theta1[theta_i][theta_j] * X[i][theta_j - 1]);
			}
		}
		sigmoid(a2, T1rows);

		// aktivacija tretjega nivoja a3
		for (int theta_i = 0; theta_i < T2rows; theta_i++) {
			a3[theta_i] = Theta2[theta_i][0]; //v bistvu moram vektorju xi dodati enico na zacetek tako da je prvi produkt enak prvi theti v vrstici.
			for (int theta_j = 1; theta_j < (T2cols + 1); theta_j++) {
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

double costFunction2(double * grad, double * param, int paramSize, double ** X, int Xrows, int Xcols, int hiddenLayerSize, double ** Y, int yLabels, double lambda) {
	/*
	Poracuna cost function in gradient parametrov, ki je uporabljen v optimizaciji
	*/
	// reshape parameters
	int iparam = 0;
	int T1rows = hiddenLayerSize;
	int T1cols = Xcols + 1;
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

	int numExamples = Xrows;

	// Vmesni in koncna matrika za feedforward
	double ** z2Matrix = allocateMatrix(T1rows, numExamples);
	double ** a2Matrix = allocateMatrix(T1rows, numExamples);
	double ** z2MatrixGradient = allocateMatrix(T1rows, numExamples);
	double ** a3Matrix = allocateMatrix(T2rows, numExamples);

	// Matriki potrebni za backpropagation
	double ** d2Matrix = allocateMatrix(T1rows, numExamples);
	double ** d3Matrix = allocateMatrix(T2rows, numExamples);


	// FEEDFORWARD
	for (int theta_i = 0; theta_i < T1rows; theta_i++) {
		for (int example_i = 0; example_i < numExamples; example_i++) {
			z2Matrix[theta_i][example_i] = T1[theta_i][0];
			for (int theta_j = 1; theta_j < (T1cols); theta_j++) {
				z2Matrix[theta_i][example_i] += (T1[theta_i][theta_j] * X[example_i][theta_j - 1]);
			}
		}
	}

	for (int i = 0; i < T1rows; i++) {
		memcpy(a2Matrix[i], z2Matrix[i], numExamples * sizeof(double));
		memcpy(z2MatrixGradient[i], z2Matrix[i], numExamples * sizeof(double));
	}
	

	//Tu zracunam sigmoido z2Matrix in sigmoid gradienta, kar rabim pri backpropagation
	for (int example_i = 0; example_i < numExamples; example_i++) {
		sigmoid2(a2Matrix, T1rows, example_i);
		sigmoidGradient2(z2MatrixGradient, T1rows, example_i);
	}


	for (int theta_i = 0; theta_i < T2rows; theta_i++) {
		for (int example_i = 0; example_i < numExamples; example_i++) {
			a3Matrix[theta_i][example_i] = T2[theta_i][0];
			for (int theta_j = 1; theta_j < (T2cols); theta_j++) {
				a3Matrix[theta_i][example_i] += (T2[theta_i][theta_j] *a2Matrix[theta_j - 1][example_i]);
			}
		}
	}

	for (int example_i = 0; example_i < numExamples; example_i++) {
		double * a3 = (double*)calloc(T1rows, sizeof(double));
		for (int i = 0; i < T2rows; i++) {
			a3[i] = a3Matrix[i][example_i];
		}
		sigmoid(a3, T2rows);
		for (int i = 0; i < T2rows; i++) {
			a3Matrix[i][example_i] = a3[i];
		}
		free(a3);
	}
	// Konec feedforward

	//pristej k cost function
	double cost = 0;
	for (int t = 0; t < numExamples; t++) {
		for (int i = 0; i < T2rows; i++) {
			cost += ((-1.0) * Y[t][i] * log(a3Matrix[i][t])) - ((1.0 - Y[t][i]) * log(1.0 - a3Matrix[i][t]));
		}
	}

	// BACKPROPAGATION
	for (int t = 0; t < numExamples; t++) {
		for (int i = 0; i < T2rows; i++) {
			d3Matrix[i][t] = a3Matrix[i][t] - Y[t][i];
		}

		for (int i = 0; i < T1rows; i++) {
			for (int j = 0; j < T2rows; j++) {
				d2Matrix[i][t] += T2[j][i + 1] * d3Matrix[j][t];
			}
			d2Matrix[i][t] = d2Matrix[i][t] * z2MatrixGradient[i][t];
		}

		for (int i = 0; i < T1rows; i++) {
			T1grad[i][0] += d2Matrix[i][t];
			for (int j = 1; j < T1cols; j++) {
				T1grad[i][j] += d2Matrix[i][t] * X[t][j - 1];
			}
		}
		for (int i = 0; i < T2rows; i++) {
			T2grad[i][0] += d3Matrix[i][t];
			for (int j = 1; j < T2cols; j++) {
				T2grad[i][j] += d3Matrix[i][t] * z2Matrix[j - 1][t];
			}
		}

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
	
	freeMatrix(z2Matrix, T1rows);
	freeMatrix(a2Matrix, T1rows);
	freeMatrix(z2MatrixGradient, T1rows);
	freeMatrix(a3Matrix, T2rows);
	freeMatrix(d2Matrix, T1rows);
	freeMatrix(d3Matrix, T2rows);
	freeMatrix(T1, T1rows);
	freeMatrix(T2, T2rows);
	freeMatrix(T1grad, T1rows);
	freeMatrix(T2grad, T2rows);
	return cost;
}

double costFunction(double * grad, double * param, int paramSize, double ** X, int Xrows, int Xcols, int hiddenLayerSize, double ** Y, int yLabels, double lambda) {
	/*
	Poracuna cost function in gradient parametrov, ki je uporabljen v optimizaciji
	*/

	// reshape parameters

	int iparam = 0;
	int T1rows = hiddenLayerSize;
	int T1cols = Xcols + 1;
	double ** T1 = allocateMatrix(T1rows, T1cols);

	int T2rows = yLabels;
	int T2cols = hiddenLayerSize + 1;
	double ** T2 = allocateMatrix(T2rows, T2cols);

	double ** T1grad = allocateMatrix(T1rows, T1cols);
	double ** T2grad = allocateMatrix(T2rows, T2cols);

	double *** T1grad_private;
	double *** T2grad_private;

	int numExamples = Xrows;

    double cost = 0;
    double costReg = 0.0;

	#pragma omp parallel
	{
		const int nthreads = omp_get_num_threads();
	    const int ithread = omp_get_thread_num();

		#pragma omp single
		{
			// vsaka nit ma svojo matriko, plast v kocki
			T1grad_private = (double***)calloc(nthreads,sizeof(double**));
			T2grad_private = (double***)calloc(nthreads,sizeof(double**));
			for(int i = 0; i < nthreads ; i++){
				T1grad_private[i] = (double **)calloc( T1rows, sizeof(double*));
				T2grad_private[i] = (double **)calloc(T2rows, sizeof(double*));
				for(int j = 0; j<T1rows; j++){
					T1grad_private[i][j] = (double*)calloc(T1cols, sizeof(double));
				}
				for(int j = 0; j<T2rows; j++){
					T2grad_private[i][j] = (double*)calloc(T2cols, sizeof(double));
				}
			}
			
			for (int j = 0; j < T1cols; j++) {
				#pragma ivdep
				#pragma vector nontemporal
				for (int i = 0; i < T1rows; i++) {
					T1[i][j] = param[iparam];
					#pragma omp atomic
					iparam++;
				}
			}
		
			for (int j = 0; j < T2cols; j++) {
				#pragma ivdep
				#pragma vector nontemporal
				for (int i = 0; i < T2rows; i++) {
					T2[i][j] = param[iparam];
					#pragma omp atomic
					iparam++;
				}
			}
		}

		#pragma omp for schedule(static) reduction(+:cost)
		// zanka cez vse primere
		for (int t = 0; t < numExamples; t++) {
			double * z2 = (double*)_mm_malloc(T1rows* sizeof(double),64);
			double * a2 = (double*)_mm_malloc(T1rows*  sizeof(double),64);
			double * a3 = (double*)_mm_malloc(T2rows* sizeof(double),64);

			memset(z2,0,T1rows * sizeof(double));
			memset(a2,0,T1rows * sizeof(double));
			memset(a3,0,T2rows * sizeof(double));

			// feedforward
			for (int theta_i = 0; theta_i < T1rows; theta_i++) {
				z2[theta_i] = T1[theta_i][0];
				#pragma ivdep
				for (int theta_j = 1; theta_j < (T1cols); theta_j++) {
					z2[theta_i] += (T1[theta_i][theta_j] * X[t][theta_j - 1]);
				}
			}
			memcpy(a2, z2, T1rows * sizeof(double));
			sigmoid(a2, T1rows);

			for (int theta_i = 0; theta_i < T2rows; theta_i++) {
				a3[theta_i] = T2[theta_i][0];
				#pragma ivdep
				for (int theta_j = 1; theta_j < (T2cols); theta_j++) {
					a3[theta_i] += (T2[theta_i][theta_j] * a2[theta_j - 1]);
				}
			}
			sigmoid(a3, T2rows);

			// pristej k cost function
			#pragma ivdep
			for (int i = 0; i < T2rows; i++) {
				cost += ((-1.0) * Y[t][i] * log(a3[i])) - ((1.0 - Y[t][i]) * log(1.0 - a3[i]));
			}

			// backpropagation
			double * d3 = (double*)_mm_malloc(T2rows* sizeof(double), 64);
			double * d2 = (double*)_mm_malloc(T1rows* sizeof(double),64);
			memset(d3,0,sizeof(double)*T2rows);
			memset(d2,0,sizeof(double)*T1rows);

			#pragma ivdep
		#pragma vector nontemporal
		for (int i = 0; i < T2rows; i++) {
			d3[i] = a3[i] - Y[t][i];
		}

		sigmoidGradient(z2, T1rows);
		for (int i = 0; i < T1rows; i++) {
			#pragma ivdep
			for (int j = 0; j < T2rows; j++) {
				d2[i] += T2[j][i + 1] * d3[j];
			}
			d2[i] = d2[i] * z2[i];
		}

		// pristej gradientu
		for (int i = 0; i < T1rows; i++) {
			T1grad_private[ithread][i][0] += d2[i];
			#pragma ivdep
			for (int j = 1; j < T1cols; j++) {
				T1grad_private[ithread][i][j] += d2[i] * X[t][j - 1];
			}
		}
		for (int i = 0; i < T2rows; i++) {
			T2grad_private[ithread][i][0] += d3[i];
			#pragma ivdep
			for (int j = 1; j < T2cols; j++) {
				T2grad_private[ithread][i][j] += d3[i] * a2[j - 1];
			}
		}

		_mm_free(z2);
		_mm_free(a2);
		_mm_free(a3);
		_mm_free(d3);
		_mm_free(d2);
	}
	

	#pragma omp for
	for (int i = 0; i < T1rows; i++) {
		for (int j = 0; j < T1cols; j++) {
			#pragma ivdep
			for(int n=0;n<nthreads;n++){
				T1grad[i][j] += T1grad_private[n][i][j];
			}
		}
	}
	#pragma omp for
	for (int i = 0; i < T2rows; i++) {
		for (int j = 0; j < T2cols; j++) {
			#pragma ivdep
			for(int n=0;n<nthreads;n++){
				T2grad[i][j] += T2grad_private[n][i][j];
			}
			
		}
	}

	//free 
	#pragma omp single
	{
		for(int n = 0; n<nthreads ;n ++){
			for(int i=0;i<T1rows;i++){
				free(T1grad_private[n][i]);
			}
			free(T1grad_private[n]);
			for(int i=0;i<T2rows;i++){
				free(T2grad_private[n][i]);
			}
			free(T2grad_private[n]);
		}
		free(T1grad_private);
		free(T2grad_private);
	}


	// normalizacija cost funkcije
	#pragma omp single
	cost = (1.0 / (double)numExamples) * cost;

	// regularizacija cost funkcije
	#pragma omp for reduction(+:costReg)
	for (int i = 0; i < T1rows; i++) {
		#pragma ivdep
		for (int j = 1; j < T1cols; j++) {
			costReg += T1[i][j] * T1[i][j];
		}
	}
	#pragma omp for reduction(+:costReg)
	for (int i = 0; i < T2rows; i++) {
		#pragma ivdep
		for (int j = 1; j < T2cols; j++) {
			costReg += T2[i][j] * T2[i][j];
		}
	}

	#pragma omp single
	cost += (lambda / (2 * (double)numExamples)) * costReg;

	// normalizacija gradienta

	#pragma omp for
	for (int i = 0; i < T1rows; i++) {
		#pragma ivdep
		#pragma vector nontemporal
		for (int j = 0; j < T1cols; j++) {
			T1grad[i][j] = T1grad[i][j] / (double)numExamples + (lambda / (double)numExamples) * T1[i][j];;
		}
	}
	#pragma omp for
	for (int i = 0; i < T2rows; i++) {
		for (int j = 0; j < T2cols; j++) {
			#pragma ivdep
			#pragma vector nontemporal
			T2grad[i][j] = T2grad[i][j] / (double)numExamples + (lambda / (double)numExamples) * T2[i][j];
		}
	}

}

// unroll gradientov
int igrad = 0;
for (int j = 0; j < T1cols; j++) {
	#pragma ivdep
	#pragma vector nontemporal
	for (int i = 0; i < T1rows; i++) {
		grad[igrad] = T1grad[i][j];
		igrad++;
	}
}
for (int j = 0; j < T2cols; j++) {
	#pragma ivdep
	#pragma vector nontemporal
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

		#pragma omp parallel for schedule(static)
		for (int i = 0; i < paramSize; i++) {
			param[i] = param[i] - alpha * grad[i];
		}

		printf("Iteration: %4d | Alpha:  %.4f | Cost: %4.6f\n", iter + 1, alpha, cost);
		fflush(stdout);
	}

	free(grad);
	freeMatrix(Y, Xrows);
	return cost;
}
