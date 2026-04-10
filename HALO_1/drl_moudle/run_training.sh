#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc_rl

if [[ -f ../install/setup.bash ]]; then
	source ../install/setup.bash
fi

python train_ppo.py "$@"
