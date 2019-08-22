#ifndef HELPERS_H_
#define HELPERS_H_

void * errexit(const char *errMessage);
void freeMatrix(double ** matrika, int rows);
double ** allocateMatrix(int rows, int cols);

#endif
