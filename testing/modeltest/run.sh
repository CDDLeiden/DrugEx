#!/bin/bash

DRUGEX_ENV=drugex
INPUT_FILE='chembl_30_100000.smi'

qsub -v MODEL=rnn,CONDA_ENV=$DRUGEX_ENV,INPUT_FILE=$INPUT_FILE,EXPERIMENT_ID=`git rev-parse --short HEAD`,WORKDIR=`pwd` qsub.sh
qsub -v MODEL=seq_trans,CONDA_ENV=$DRUGEX_ENV,INPUT_FILE=$INPUT_FILE,EXPERIMENT_ID=`git rev-parse --short HEAD`,WORKDIR=`pwd` qsub.sh
qsub -v MODEL=graph,CONDA_ENV=$DRUGEX_ENV,INPUT_FILE=$INPUT_FILE,EXPERIMENT_ID=`git rev-parse --short HEAD`,WORKDIR=`pwd` qsub.sh
