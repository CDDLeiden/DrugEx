#!/bin/bash

qsub -v MODEL=rnn,CONDA_ENV=drugex,EXPERIMENT_ID=`git rev-parse --short HEAD`,WORKDIR=`pwd` qsub.sh
qsub -v MODEL=seq_trans,CONDA_ENV=drugex,EXPERIMENT_ID=`git rev-parse --short HEAD`,WORKDIR=`pwd` qsub.sh
qsub -v MODEL=graph,CONDA_ENV=drugex,EXPERIMENT_ID=`git rev-parse --short HEAD`,WORKDIR=`pwd` qsub.sh
