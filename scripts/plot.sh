#!/bin/sh

set -xe

./scripts/plot_memory.py < "$1"
./scripts/plot_performance.py < "$1"