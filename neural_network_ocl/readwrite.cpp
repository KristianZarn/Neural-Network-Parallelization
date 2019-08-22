#include <stdio.h>
#include <stdlib.h>

#include "readwrite.h"
#include "helpers.h"

float ** readMatrixFromFile(const char * path, int rows, int cols) {
	/*
	Prebere matriko iz datoteke (stevila morajo biti locena s presledki)
	*/

	float ** matrika = (float**)malloc(sizeof(float*) * rows);
	FILE * f;
	f = fopen(path, "r");
	if (f == NULL) errexit("Napaka pri branju datoteke");

	for (int i = 0; i < rows; i++) {
		matrika[i] = (float*)malloc(sizeof(float) * cols);
		for (int j = 0; j < cols; j++) {
			if(fscanf(f, "%f", &(matrika[i][j])) == -1) errexit("Napaka fscanf pri branju");
		}
	}
	fclose(f);
	return matrika;
}

float * readVectorFromFile(const char * path, int elements) {
	/*
	Funkcija prebere vektor stevil iz datoteke
	*/

	float * vektor = (float*)malloc(sizeof(float)*elements);
	FILE * f;
	f = fopen(path, "r");
	if (f == NULL) errexit("Napaka pri branju datoteke");

	for (int i = 0; i < elements; i++) {
		if(fscanf(f, "%f", &(vektor[i])) == -1) errexit("Napaka fscanf pri branju");
	}
	fclose(f);
	return vektor;
}

void writeMatrixToFile(const char * path, float ** matrix, int rows, int cols) {
	/*
	Zapise matriko stevil v datoteko
	*/

	FILE * f;
	f = fopen(path, "w");
	if (f == NULL) errexit("Napaka pri pisanju v  datoteko");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			fprintf(f, "%f\t", matrix[i][j]);
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

void writeVectorToFile(const char * path, float * vector, int elements) {
	/*
	Zapise vektor stevil v datoteko
	*/

	FILE * f;
	f = fopen(path, "w");
	if (f == NULL) errexit("Napaka pri pisanju datoteke");

	for (int i = 0; i < elements; i++) {
		fprintf(f, "%f\n", vector[i]);
	}
	fclose(f);
}

void writeParametersToFile(const char * pathT1, const char * pathT2, float * param, int Xcols, int hiddenLayerSize, int yLabels) {
	/*
	Vektor parametrov preoblikuje v matriki in ju zapise v datoteko
	*/

	int iparam = 0;
	int T1rows = hiddenLayerSize;
	int T1cols = Xcols + 1;
	float ** T1 = allocateMatrix(T1rows, T1cols);
	for (int j = 0; j < T1cols; j++) {
		for (int i = 0; i < T1rows; i++) {
			T1[i][j] = param[iparam];
			iparam++;
		}
	}
	int T2rows = yLabels;
	int T2cols = hiddenLayerSize + 1;
	float ** T2 = allocateMatrix(T2rows, T2cols);
	for (int j = 0; j < T2cols; j++) {
		for (int i = 0; i < T2rows; i++) {
			T2[i][j] = param[iparam];
			iparam++;
		}
	}

	writeMatrixToFile(pathT1, T1, T1rows, T1cols);
	writeMatrixToFile(pathT2, T2, T2rows, T2cols);

	freeMatrix(T1, T1rows);
	freeMatrix(T2, T2rows);
}
