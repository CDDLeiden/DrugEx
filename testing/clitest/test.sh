#!/bin/bash

## Test some common command line options automatically.

source env.sh

cleanup

./test_single.sh
./test_graph.sh
./test_gpt.sh
echo "Cleaning up..."
cleanup
echo "Done."
