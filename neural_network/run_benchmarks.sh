#!/bin/bash

# Uporaba:
#./benchmark [totalRuns] [datasetSize] [hiddenLayerSize] [iterations]

RUNS=5

DATASET=10000
./benchmark ${RUNS} ${DATASET} 25 50
./benchmark ${RUNS} ${DATASET} 25 100
./benchmark ${RUNS} ${DATASET} 25 150
./benchmark ${RUNS} ${DATASET} 25 200
