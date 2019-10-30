#!/bin/bash

dataset="satimage"
lambda_max=1e+7
lambda_min=1
k=15
BOUND="RRPB"


INT_MAX=2147483647
EPS=1e-6
FREQ=10
COUNT=5
TRAIN=0.6

./run.exe $dataset $k $lambda_max $lambda_min $FREQ $BOUND $EPS $TRAIN $COUNT

