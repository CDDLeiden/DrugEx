#!/bin/bash

export TIMESTAMP=$(date +%Y%m%d%H%M%S)

docker-compose rm -f
echo "Spawning Runner... TIMESTAMP=${TIMESTAMP}"
mkdir -p logs
docker-compose config >> "logs/${TIMESTAMP}.log"
docker-compose up --force-recreate
docker-compose logs >> "logs/${TIMESTAMP}.log"
echo "Logs saved in logs/${TIMESTAMP}.log"
echo "Cleaning up..."
docker-compose rm -f
docker image rm drugex-test-runner-${TIMESTAMP}
