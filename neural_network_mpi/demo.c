#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#include "helpers.h"
#include "readwrite.h"
#include "neuralnetwork.h"

#define BILLION 1E9

void demo() {
	int procs, myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &procs);
//	printf("Hi, I am process %d\n", myid);
	double tic, toc;	
	//if(myid == 0){ // samo root dela to na zacetku

	
	double duration;

	// vhodni podatki
	if(myid == 0)printf("\nReading input data... \n");
	// ucna mnozica
	int Xrows = 10000;
	int Xcols = 28 * 28;
	double ** X = readMatrixFromFile("../MNIST/train-images-10k.dat", Xrows, Xcols);

	int yLabels = 10;
	double * y = readVectorFromFile("../MNIST/train-labels-10k.dat", Xrows);

	// testna mnozica
	int Xtestrows = 10000;
	double ** Xtest = readMatrixFromFile("../MNIST/test-images-10k.dat", Xtestrows, Xcols);
	double * ytest = readVectorFromFile("../MNIST/test-labels-10k.dat", Xtestrows);
	if(myid == 0)printf("DONE\n");

	// UCENJE
	int hiddenLayerSize = 25;
	double lambda = 0.1;
	int iterations = 20;

	int paramSize = hiddenLayerSize * (Xcols + 1) + yLabels * (hiddenLayerSize + 1);
	double * param = (double*)calloc(paramSize, sizeof(double));
	randInitializeWeights(param, paramSize);
	//debugInitializeWeights(param, paramSize);

	if(myid == 0)printf("\nTraining Neural Network... \n");
//	MPI_Barrier(MPI_COMM_WORLD);
	tic = MPI_Wtime();
	double cost = gradientDescent(param, paramSize, iterations, X, Xrows, Xcols, hiddenLayerSize, y, yLabels, lambda);
	toc = MPI_Wtime();

	if(myid == 0) {
		//clock_gettime(CLOCK_MONOTONIC,&toc);
		duration = toc - tic;
		printf("Training took: %.2lfs \n", duration);
		printf("DONE\n");

	    // PREDIKCIJA
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

    		double * result = predict(Xtest, Xtestrows, Xcols, T1, T1rows, T1cols, T2, T2rows, T2cols);

    		int correct = 0;
    		for (int i = 0; i < Xtestrows; i++) {
        		if (ytest[i] == result[i]) correct++;
    		}
   		double accuracy = ((double)correct / (double)Xtestrows) * 100.0;
    		printf("\nTraining Set Accuracy:\n");
    		printf("Accuracy: %.3f%%\n", accuracy);
    		
		free(result);
    		freeMatrix(T1, T1rows);
		freeMatrix(T2, T2rows);
	
	}
    free(param);

	// ciscenje
	freeMatrix(X, Xrows);
	free(y);
	freeMatrix(Xtest, Xtestrows);
	free(ytest);
	if(myid == 0)printf("DONE\n");
}

int main() {
	//srand((unsigned int)time(NULL));
	MPI_Init(NULL,NULL);
	srand(1);

	demo();
	MPI_Finalize();	
	return 0;
}
