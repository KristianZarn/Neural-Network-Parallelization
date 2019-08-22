#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "helpers.h"
#include "readwrite.h"
#include "neuralnetwork_omp.h"

#define BILLION 1E9

void demo() {
    struct timespec tic, toc;
    double duration;

    // vhodni podatki
    printf("\nReading input data... \n");
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
    printf("DONE\n");

    // UCENJE
    int hiddenLayerSize = 25;
    double lambda = 0.1;
    int iterations = 20;

    int paramSize = hiddenLayerSize * (Xcols + 1) + yLabels * (hiddenLayerSize + 1);
    double * param = (double*)calloc(paramSize, sizeof(double));
    randInitializeWeights(param, paramSize);
    //debugInitializeWeights(param, paramSize);

    printf("\nTraining Neural Network... \n");
    clock_gettime(CLOCK_MONOTONIC,&tic);
    double cost = gradientDescent(param, paramSize, iterations, X, Xrows, Xcols, hiddenLayerSize, y, yLabels, lambda);
    clock_gettime(CLOCK_MONOTONIC,&toc);
    duration = ( toc.tv_sec - tic.tv_sec ) + ( toc.tv_nsec - tic.tv_nsec ) / BILLION;
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

    free(param);
    freeMatrix(T1, T1rows);
    freeMatrix(T2, T2rows);
    free(result);

    // ciscenje
    freeMatrix(X, Xrows);
    free(y);
    freeMatrix(Xtest, Xtestrows);
    free(ytest);
    printf("DONE\n");
}

int main() {
    //srand((unsigned int)time(NULL));
    srand(1);

    int threadsMax = omp_get_max_threads();
    printf("Max number of threads = %d\n", threadsMax);

    demo();
    return 0;
}
