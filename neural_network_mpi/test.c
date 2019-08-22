#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>


int main(){
	int procs, myid;
	int*sendcounts; // tabela prikazuje st. elementov za vsak proces
	int*displs; // tabela prikazuje indeks elementa, kjer se zacne vsaka podtabela za proces
	int scatterSize, remainder;
	int displId = 0; // s tem racunam indeks displs
	int Xcols = 28*28;
	int Xrows = 10000;

	MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	

	sendcounts = (int*)malloc(sizeof(int)*procs);
	displs = (int*)malloc(sizeof(int)*procs);

	double * _X = (double*)malloc(sizeof(double)*Xcols*Xrows);
	int counter=0;
	for(int i=0;i<Xrows;i++){
		for(int j=0;j<Xcols;j++){
			_X[counter]=2;
			counter++;
		}
	}

	scatterSize = (Xrows / procs) * Xcols;
	remainder = Xrows % procs; // ostalo stevilo vrstic
	printf("scatterSize: %d, remainder: %d\n", scatterSize, remainder);

	for(int i=0;i<procs;i++){
		sendcounts[i]=scatterSize;
		if(remainder>0){
			sendcounts[i]+= Xcols;
			remainder-=1;
		}
		displs[i]=displId;
		displId+=sendcounts[i];
		printf("sendcounts[%d],%d",i,sendcounts[i]);
	}

	//double ** myX = (double**)malloc(sizeof(double)*sendcounts[myid]/Xcols);
	double * myX = (double*)malloc(sizeof(double)*sendcounts[0]);

	//for(int i=0; i < sendcounts[myid] / Xcols ; i++){
	//	myX[i] = (double*)malloc(sizeof(Xcols));
	//}

	if(myid==0){
  		for(int i=0;i<procs;i++){
  			printf("sendcounts[%d]: %d\tdispls[%d]=%d\n",i,sendcounts[i],i,displs[i]);
  		}
  	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Scatterv(_X,sendcounts,displs,MPI_DOUBLE,myX,sendcounts[myid], MPI_DOUBLE,0,MPI_COMM_WORLD);
	printf("pred cost function process %d\n", myid);

	MPI_Finalize();
return 0;
}
