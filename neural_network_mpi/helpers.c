#include <stdlib.h>
#include <stdio.h>

#include "helpers.h"

void * errexit(const char *errMessage) {
	/*
	Prikaz sporocila napake
	*/
	perror(errMessage);
	exit(-1);
}

void freeMatrix(double ** matrika, int rows) {
	/*
	Funkcija sprosti pointerje dvodimenzionalne tabele
	*/

	for (int i = 0; i < rows; i++) {
		free(matrika[i]);
	}
	free(matrika);
}

double ** allocateMatrix(int rows, int cols) {
	/*
	Rezervira prostor za matriko
	*/

	double ** matrix = (double**)calloc(rows, sizeof(double*));
	for (int i = 0; i < rows; i++) {
		matrix[i] = (double*)calloc(cols, sizeof(double));
	}
	return matrix;
}
