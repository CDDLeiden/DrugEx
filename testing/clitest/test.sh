#!/bin/bash

## Test some common command line options for all models.

source env.sh

cleanup

./test_seq_rnn.sh
./test_graph_trans.sh
./test_seq_trans.sh

cleanup
