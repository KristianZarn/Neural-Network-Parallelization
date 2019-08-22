#!/bin/bash

# Uporaba:
#./benchmark [totalRuns] [datasetSize] [hiddenLayerSize] [iterations]

RUNS=5

./benchmark ${RUNS} 5000 25 100
./benchmark ${RUNS} 10000 25 100
./benchmark ${RUNS} 15000 25 100
./benchmark ${RUNS} 20000 25 100
./benchmark ${RUNS} 25000 25 100

