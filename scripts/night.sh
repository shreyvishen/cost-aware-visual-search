#!/usr/bin/env bash
# The unattended night. Runs A, secures it, then runs B. Idempotent and crash-resistant:
# a run that already wrote DONE is skipped, and a crashed run is resumed from its last
# checkpoint up to MAX_TRIES times before we give up and move on to protect the next run.
set -u
cd /srv/ai/code/cost-aware-visual-search
PY=/home/shreyv/venvs/vllm-ocr/bin/python
export CUDA_VISIBLE_DEVICES=2,3,4
# Fragmentation, not capacity, is what kills a long multi-image run on 24 GB cards.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAX_TRIES=3
RETRY_SLEEP=60

# Never start on a card someone else (or a zombie of ours) still holds. A previous restart
# OOMed because the outgoing process had not released its 17 GiB yet.
wait_for_free_gpus () {
  for i in $(seq 1 30); do
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
           | awk -F', ' '$1==2 || $1==3 || $1==4 {gsub(" MiB","",$2); if ($2+0 > 2000) print $1}')
    if [ -z "$busy" ]; then return 0; fi
    echo "[night] waiting for GPUs to free: $(echo $busy | tr '\n' ' ')"
    sleep 10
  done
  echo "[night] WARNING: GPUs still busy after 5 min; starting anyway"
}

run_one () {
  local cfg=$1 out=$2 name=$3
  if [ -f "$out/DONE" ]; then echo "[night] $name already DONE, skipping"; return 0; fi
  for try in $(seq 1 $MAX_TRIES); do
    wait_for_free_gpus
    echo "[night] === $name attempt $try $(date '+%F %T') ==="
    $PY -m src.train --config "$cfg" --out "$out" --resume >> "$out.log" 2>&1
    rc=$?
    echo "[night] $name attempt $try exited rc=$rc"
    if [ -f "$out/DONE" ]; then echo "[night] $name DONE"; return 0; fi
    echo "[night] $name will retry in ${RETRY_SLEEP}s (resuming from last checkpoint)"
    sleep $RETRY_SLEEP
  done
  echo "[night] $name FAILED after $MAX_TRIES attempts — moving on to protect the next run"
  return 1
}

mkdir -p /srv/ai/runs
run_one configs/run_a.json /srv/ai/runs/run_a A
echo "[night] A finished at $(date '+%F %T'); artifacts:"
du -sh /srv/ai/runs/run_a 2>/dev/null
# The Mac pulls every 60 s; give it a full cycle to secure A before B touches the GPUs.
sleep 75
run_one configs/run_b.json /srv/ai/runs/run_b B
echo "[night] B finished at $(date '+%F %T')"
du -sh /srv/ai/runs/run_b 2>/dev/null
echo "[night] ALL DONE $(date '+%F %T')"
