#! /usr/bin/bash

PY3="python"
CONFIG_BASE_DIR="configs/modelnet40/pvcnn/experiments"
# ${PY3} train.py --config ${CONFIG_BASE_DIR}/z_SO3/exp8.py --device 1,2
# ${PY3} train.py --config ${CONFIG_BASE_DIR}/z_SO3/exp9.py --device 1,2
${PY3} train.py --config ${CONFIG_BASE_DIR}/SO3_SO3/exp13.py --device 0
# --evaluate