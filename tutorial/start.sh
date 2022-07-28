#!/bin/bash

stdbuf -oL nohup jupyter-lab --no-browser --port=${1:-8888} > notebook.log 2>&1 &
echo "$!" > pid.log
