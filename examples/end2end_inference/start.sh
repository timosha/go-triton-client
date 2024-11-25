#!/bin/bash
set -ex

tritonserver \
  --model-repository=/models \
  --load-model=ty_bert \
  --load-model=ty_roberta \
  --model-control-mode=explicit \
  --allow-gpu-metrics=true \
  --allow-metrics=true &

TRITON_PID=$!

function wait_for_triton() {
  echo "Waiting..."
  until [ $(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v2/health/ready) -eq 200 ]; do
    sleep 1
  done
  echo "Triton ready!"
}

wait_for_triton

echo "Starting application..."
/main || {
  echo "Error code: $?"
  exit 1
}

wait $TRITON_PID