#ifndef READWRITE_H_
#define READWRITE_H_

float ** readMatrixFromFile(const char * path, int rows, int cols);
float * readVectorFromFile(const char * path, int elements);
void writeMatrixToFile(const char * path, float ** matrix, int rows, int cols);
void writeVectorToFile(const char * path, float * vector, int elements);
void writeParametersToFile(const char * pathT1, const char * pathT2, float * param, int Xcols, int hiddenLayerSize, int yLabels);

#endif
