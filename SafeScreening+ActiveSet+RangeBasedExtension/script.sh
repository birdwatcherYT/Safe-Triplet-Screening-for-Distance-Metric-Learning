#!/bin/bash

dataset="wine"
lambda_max=2e+7
k=999999


INT_MAX=2147483647
EPS=1e-6
FREQ=10
COUNT=5
TRAIN=0.9

./run.exe $dataset $k $lambda_max $FREQ "ALL" $EPS $TRAIN $COUNT

