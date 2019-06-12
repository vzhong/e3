#!/usr/bin/env bash
set -x
nvidia-docker run --rm \
  -v $PWD:/opt/code \
  -u $(id -u $USER):$(id -g $USER) \
  vzhong/e3 \
  $@
