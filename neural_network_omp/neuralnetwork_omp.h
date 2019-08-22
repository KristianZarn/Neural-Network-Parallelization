#ifndef NEURALNETWORK_PTHREADS_H_
#define NEURALNETWORK_PTHREADS_H_

double * predict(double ** X, int Xrows, int Xcols, double ** Theta1, int T1rows, int T1cols, double ** Theta2, int T2rows, int T2cols);
void randInitializeWeights(double * param, int paramSize);
void debugInitializeWeights(double * param, int paramSize);
double gradientDescent(double * param, int paramSize, int iterations, double ** X, int Xrows, int Xcols, int hiddenLayerSize, double * y, int yLabels, double lambda);

#endif
