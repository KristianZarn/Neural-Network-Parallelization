#ifndef READWRITE_H_
#define READWRITE_H_

double ** readMatrixFromFile(const char * path, int rows, int cols);
double * readVectorFromFile(const char * path, int elements);
void writeMatrixToFile(const char * path, double ** matrix, int rows, int cols);
void writeVectorToFile(const char * path, double * vector, int elements);
void writeParametersToFile(const char * pathT1, const char * pathT2, double * param, int Xcols, int hiddenLayerSize, int yLabels);

#endif
