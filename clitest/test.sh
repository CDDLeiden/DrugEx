#!/bin/bash

## Test some common command line options automatically.

source env.sh

./test_graph.sh
./test_gpt.sh
./test_ved.sh
./test_attn.sh
./test_single.sh

cleanup
