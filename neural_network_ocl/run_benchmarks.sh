#!/bin/bash

# Uporaba:
#./benchmark [totalRuns] [datasetSize] [hiddenLayerSize] [iterations]

RUNS=10

DATASET=10000
./benchmark ${RUNS} ${DATASET} 25 50
./benchmark ${RUNS} ${DATASET} 25 100
./benchmark ${RUNS} ${DATASET} 25 150
./benchmark ${RUNS} ${DATASET} 25 200

DATASET=20000
./benchmark ${RUNS} ${DATASET} 25 50
./benchmark ${RUNS} ${DATASET} 25 100
./benchmark ${RUNS} ${DATASET} 25 150
./benchmark ${RUNS} ${DATASET} 25 200

DATASET=30000
./benchmark ${RUNS} ${DATASET} 25 50
./benchmark ${RUNS} ${DATASET} 25 100
./benchmark ${RUNS} ${DATASET} 25 150
./benchmark ${RUNS} ${DATASET} 25 200

DATASET=40000
./benchmark ${RUNS} ${DATASET} 25 50
./benchmark ${RUNS} ${DATASET} 25 100
./benchmark ${RUNS} ${DATASET} 25 150
./benchmark ${RUNS} ${DATASET} 25 200

DATASET=50000
./benchmark ${RUNS} ${DATASET} 25 50
./benchmark ${RUNS} ${DATASET} 25 100
./benchmark ${RUNS} ${DATASET} 25 150
./benchmark ${RUNS} ${DATASET} 25 200

DATASET=60000
./benchmark ${RUNS} ${DATASET} 25 50
./benchmark ${RUNS} ${DATASET} 25 100
./benchmark ${RUNS} ${DATASET} 25 150
./benchmark ${RUNS} ${DATASET} 25 200
