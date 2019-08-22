#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

float * predict(float ** X, int Xrows, int Xcols, float ** Theta1, int T1rows, int T1cols, float ** Theta2, int T2rows, int T2cols);
void randInitializeWeights(float * param, int paramSize);
float gradientDescent(float * param, int paramSize, int iterations, float ** X, int Xrows, int Xcols, int hiddenLayerSize, float * y, int yLabels, float lambda);

#endif
