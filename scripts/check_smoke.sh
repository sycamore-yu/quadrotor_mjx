#!/bin/bash
set -e

# Get repo root directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$DIR")"
cd "$ROOT_DIR"

# Determine python executable
if [ -d ".venv" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python"
fi

echo "=== Running smoke training matrix ==="
$PYTHON scripts/train.py --env hover_state --algo bptt --smoke
$PYTHON scripts/train.py --env hover_state --algo ppo --smoke
$PYTHON scripts/train.py --env hover_state --algo shac --smoke

$PYTHON scripts/train.py --env hover_features --algo bptt --smoke

$PYTHON scripts/train.py --env hover_obstacle --algo bptt --smoke
$PYTHON scripts/train.py --env hover_obstacle --algo ppo --smoke
$PYTHON scripts/train.py --env hover_obstacle --algo shac --smoke

$PYTHON scripts/train.py --env gate_crossing --algo bptt --smoke
$PYTHON scripts/train.py --env gate_crossing --algo ppo --smoke
$PYTHON scripts/train.py --env gate_crossing --algo shac --smoke

$PYTHON scripts/train.py --env forest_navigation --algo bptt --smoke
$PYTHON scripts/train.py --env forest_navigation --algo ppo --smoke
$PYTHON scripts/train.py --env forest_navigation --algo shac --smoke

echo "=== All smoke checks passed successfully! ==="
