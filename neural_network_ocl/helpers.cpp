#include <stdlib.h>
#include <cstdio>

#include "helpers.h"

void * errexit(const char *errMessage) {
	/*
	Prikaz sporocila napake
	*/
	perror(errMessage);
	exit(-1);
}

void freeMatrix(float ** matrika, int rows) {
	/*
	Funkcija sprosti pointerje dvodimenzionalne tabele
	*/

	for (int i = 0; i < rows; i++) {
		free(matrika[i]);
	}
	free(matrika);
}

float ** allocateMatrix(int rows, int cols) {
	/*
	Rezervira prostor za matriko
	*/

	float ** matrix = (float**)calloc(rows, sizeof(float*));
	for (int i = 0; i < rows; i++) {
		matrix[i] = (float*)calloc(cols, sizeof(float));
	}
	return matrix;
}
