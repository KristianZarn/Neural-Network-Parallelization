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

void benchmark(int numExamples) {
	int procs, myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &procs);
	
	double tic, toc;	
	double duration;

	if(myid == 0)printf("\nReading input data... \n");
	// ucna mnozica
	int Xrows = numExamples;
	int Xcols = 28 * 28;

	// preberem training bazo
	char trainFile[100];
	snprintf(trainFile, 100, "train-images-%dk.dat", numExamples/1000);
	double ** X = readMatrixFromFile(trainFile, Xrows, Xcols);

	int yLabels = 10;
	char trainLabelsFile[100];
	snprintf(trainLabelsFile, 100, "train-labels-%dk.dat", numExamples/1000);
	double * y = readVectorFromFile(trainLabelsFile, Xrows);

	// testna mnozica
	//int Xtestrows = numExamples;
	//double ** Xtest = readMatrixFromFile("test-images-10k.dat", Xtestrows, Xcols);
	//double * ytest = readVectorFromFile("test-labels-10k.dat", Xtestrows);
	if(myid == 0)printf("DONE\n");

	int hiddenLayerSize = 25;
	double lambda = 0.1;
	int iterations = 100;

	if(myid==0){
		printf("Dataset size= %d\nHidden layer size = %d\nIterations = %d\n", 
		numExamples, hiddenLayerSize, iterations);
	}
	//debugInitializeWeights(param, paramSize);

    if(myid == 0)printf("\nTraining Neural Network... \n");
    int runs =10 ;
    double * durations = (double*)calloc(runs, sizeof(double));
	for(int i=0;i<runs;i++){
		// UCENJE

		int paramSize = hiddenLayerSize * (Xcols + 1) + yLabels * (hiddenLayerSize + 1);
		double * param = (double*)calloc(paramSize, sizeof(double));
		randInitializeWeights(param, paramSize);
		tic = MPI_Wtime();
		double cost = gradientDescent(param, paramSize, iterations, X, Xrows, Xcols, hiddenLayerSize, y, yLabels, lambda);
		toc = MPI_Wtime();
	
	     if(myid == 0) {
		//clock_gettime(CLOCK_MONOTONIC,&toc);
		duration = toc - tic;
		//printf("Training took: %.2lfs \n", duration);
		//printf("DONE\n");
		durations[i] = duration;
		
		// PREDIKCIJA
		// Odkomentiraj ce zelis tudi videti rezultate predikcije modela
		/*
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
		*/
		free(param);
	   } //myid ==0
	}//for loop
	if(myid == 0){
		double mean = 0;
	    	for (int i = 0; i < runs; i++) {
			mean += durations[i];
		}
		mean = mean / runs;

		double se = 0;
		    for (int i = 0; i < runs; i++) {
		            double tmp = durations[i] - mean;
			    tmp = tmp * tmp;
		            se += tmp;
		    }
		se = sqrt(se / runs) / sqrt(runs);
	
		printf("\nPovprecje:\n%.3f +- %.3f\n", mean, se);
	}
	// ciscenje

	freeMatrix(X, Xrows);
	free(y);
	//freeMatrix(Xtest, Xtestrows);
	//free(ytest);
	if(myid == 0)printf("DONE\n\n\n");
}

int main(int argc, char ** argv) {
	MPI_Init(NULL,NULL);
	int myid = 0;
	int procs = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &procs);
	
	//srand((unsigned int)time(NULL));
	srand(1);
	for(int i=10000; i <= 60000; i+=10000){	
		if(myid == 0)printf("Ucna mnozica: %d\n",i);
		benchmark(i);
	}
	MPI_Finalize();	
	return 0;
}
