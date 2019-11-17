#!/bin/bash
PLANNING_PROJECT_DATA="$(pwd)/../data"
export PLANNING_PROJECT_DATA
source venv/bin/activate
echo "Set environment variable PLANNING_PROJECT_DATA=${PLANNING_PROJECT_DATA}"
echo "Activated Virtual Environment."