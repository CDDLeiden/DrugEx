#!/bin/bash

## Test some common command line options automatically.

source env.sh

cleanup

./test_seq_rnn.sh
./test_graph_trans.sh
./test_seq_trans.sh
# echo "Cleaning up..."
cleanup
# echo "Done."
