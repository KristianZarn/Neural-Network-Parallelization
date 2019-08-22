#!/bin/bash

# Uporaba:
#./benchmark [totalRuns] [datasetSize] [hiddenLayerSize] [iterations]

RUNS=10
DATASET=10000

export OMP_NUM_THREADS=1
./benchmark ${RUNS} ${DATASET} 25 20
./benchmark ${RUNS} ${DATASET} 25 40
./benchmark ${RUNS} ${DATASET} 25 60
./benchmark ${RUNS} ${DATASET} 25 80
./benchmark ${RUNS} ${DATASET} 25 100

export OMP_NUM_THREADS=4
./benchmark ${RUNS} ${DATASET} 25 20
./benchmark ${RUNS} ${DATASET} 25 40
./benchmark ${RUNS} ${DATASET} 25 60
./benchmark ${RUNS} ${DATASET} 25 80
./benchmark ${RUNS} ${DATASET} 25 100

export OMP_NUM_THREADS=8
./benchmark ${RUNS} ${DATASET} 25 20
./benchmark ${RUNS} ${DATASET} 25 40
./benchmark ${RUNS} ${DATASET} 25 60
./benchmark ${RUNS} ${DATASET} 25 80
./benchmark ${RUNS} ${DATASET} 25 100

export OMP_NUM_THREADS=16
./benchmark ${RUNS} ${DATASET} 25 20
./benchmark ${RUNS} ${DATASET} 25 40
./benchmark ${RUNS} ${DATASET} 25 60
./benchmark ${RUNS} ${DATASET} 25 80
./benchmark ${RUNS} ${DATASET} 25 100

export OMP_NUM_THREADS=24
./benchmark ${RUNS} ${DATASET} 25 20
./benchmark ${RUNS} ${DATASET} 25 40
./benchmark ${RUNS} ${DATASET} 25 60
./benchmark ${RUNS} ${DATASET} 25 80
./benchmark ${RUNS} ${DATASET} 25 100

export OMP_NUM_THREADS=32
./benchmark ${RUNS} ${DATASET} 25 20
./benchmark ${RUNS} ${DATASET} 25 40
./benchmark ${RUNS} ${DATASET} 25 60
./benchmark ${RUNS} ${DATASET} 25 80
./benchmark ${RUNS} ${DATASET} 25 100
