#ifndef HELPERS_H_
#define HELPERS_H_

void * errexit(const char *errMessage);
void freeMatrix(float ** matrika, int rows);
float ** allocateMatrix(int rows, int cols);

#endif
