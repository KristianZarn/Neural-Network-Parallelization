#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

void sigmoid(double* vector, int len);
void sigmoidGradient(double* vector, int len);
double * predict(double ** X, int Xrows, int Xcols, double ** Theta1, int T1rows, int T1cols, double ** Theta2, int T2rows, int T2cols);
void randInitializeWeights(double * param, int paramSize);
void debugInitializeWeights(double * param, int paramSize);
double costFunction(double * grad, double * param, int paramSize, double * myX, int * sendcounts, int * displs, int scatterSize, int Xrows, int Xcols, int hiddenLayerSize, double ** Y, int yLabels, double lambda);
double gradientDescent(double * param, int paramSize, int iterations, double ** X, int Xrows, int Xcols, int hiddenLayerSize, double * y, int yLabels, double lambda);

#endif
