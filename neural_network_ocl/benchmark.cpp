#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "helpers.h"
#include "readwrite.h"
#include "neuralnetwork_ocl.h"

#define BILLION 1E9

void benchmark(int runs, int numExamples, int hiddenLayerSize, int iterations) {
	struct timespec StartingTime, EndingTime;
    double duration;

    // vhodni podatki
    printf("\nReading input data... \n");

    // ucna mnozica
    int Xrows = numExamples;
    int Xcols = 28 * 28;
    char trainFile[100];
    snprintf(trainFile, 100, "../MNIST/train-images-%dk.dat", numExamples/1000);
	float ** X = readMatrixFromFile(trainFile, Xrows, Xcols);

    int yLabels = 10;
    char trainLabelsFile[100];
    snprintf(trainLabelsFile, 100, "../MNIST/train-labels-%dk.dat", numExamples/1000);
	float * y = readVectorFromFile(trainLabelsFile, Xrows);

    // testna mnozica
    int Xtestrows = 10000;
	float ** Xtest = readMatrixFromFile("../MNIST/test-images-10k.dat", Xtestrows, Xcols);
	float * ytest = readVectorFromFile("../MNIST/test-labels-10k.dat", Xtestrows);
    printf("DONE\n");

    // datoteka z rezultati
    char benchmarkfile [100];
    snprintf(benchmarkfile, 100, "benchmarks/N%d-H%d-I%d.txt", numExamples, hiddenLayerSize, iterations);

    FILE * f;
	f = fopen(benchmarkfile, "w");
    if (f == NULL) errexit("Napaka pri pisanju v datoteko");

    fprintf(f, "Dataset size= %d\nHidden layer size = %d\nIterations = %d\n\n",
            numExamples, hiddenLayerSize, iterations);

    double * durations = (double*)calloc(runs, sizeof(double));
    for (int run = 0; run < runs; run++) {
        // UCENJE
		float lambda = 0.1f;
        int paramSize = hiddenLayerSize * (Xcols + 1) + yLabels * (hiddenLayerSize + 1);
		float * param = (float*)calloc(paramSize, sizeof(float));
        randInitializeWeights(param, paramSize);

        printf("\nTraining Neural Network... \n");

		clock_gettime(CLOCK_MONOTONIC,&StartingTime);
		float cost = gradientDescent(param, paramSize, iterations, X, Xrows, Xcols, hiddenLayerSize, y, yLabels, lambda);
		clock_gettime(CLOCK_MONOTONIC,&EndingTime);
		duration = ( EndingTime.tv_sec - StartingTime.tv_sec )+ ( EndingTime.tv_nsec - StartingTime.tv_nsec ) / BILLION;
        free(param);

        printf("Training took: %lfs \n", duration);
        durations[run] = duration;

        // zapisi v datoteko z rezultati
        fprintf(f, "Run:\t%4d\tDuration:\t%lf\n", run+1, duration);
    }

    // poracunaj statistike
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

    fprintf(f, "\nPovprecje:\n%.3f +- %.3f\n", mean, se);
    fclose(f);

    // ciscenje
    freeMatrix(X, Xrows);
    free(y);
    freeMatrix(Xtest, Xtestrows);
    free(ytest);
    printf("DONE\n");
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        printf("Pozenite z naslednjimi argumenti:\n");
        printf("  %s [totalRuns] [datasetSize] [hiddenLayerSize] [iterations]\n", argv[0]);
        return 0;
    }

    srand((unsigned int)time(NULL));
    //srand(1);

    // parametri iz ukazne vrstice
    int runs = strtol(argv[1], NULL, 10);
    int numExamples = strtol(argv[2], NULL, 10);
    int hiddenLayerSize = strtol(argv[3], NULL, 10);
    int iterations = strtol(argv[4], NULL, 10);

    printf("Running with parameters: \n");
    printf("Runs = %d, Num = %d, Hidden = %d, Iter = %d\n", runs, numExamples, hiddenLayerSize, iterations);

    benchmark(runs, numExamples, hiddenLayerSize, iterations);
    return 0;
}
