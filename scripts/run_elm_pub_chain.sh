#!/usr/bin/env bash
# Publication-rigour chain for elm GPU 0: A6.1-A6.3 probes then A6.4 sweep.
set -u
PY=/home_nfs/shivam/harvard/cs_2881_hw0/.venv/bin/python
S=/home_nfs/shivam/harvard/cs_2881_hw0/harvard-cs2881-hw0/scripts
echo "=== pub_probes start $(date -u +%H:%M:%S)"
$PY -u "$S/run_cot_pub_probes.py" --n 150 --max-new 512
echo "=== pub_probes rc=$?"
echo "=== maintenance_sweep start $(date -u +%H:%M:%S)"
$PY -u "$S/run_maintenance_sweep.py"
echo "=== maintenance_sweep rc=$?"
echo "ELM_PUB_CHAIN_COMPLETE $(date -u +%H:%M:%S)"
