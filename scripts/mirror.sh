#!/usr/bin/env bash
# Continuous off-machine mirror (GOAL §11/§17). The MAC PULLS from the rig, so no rig->Mac
# credentials are needed and a total rig loss costs at most one cycle of artifacts.
#
# Two things this script gets deliberately right, both learned the hard way:
#
#  1. `trainer_state.pt` is EXCLUDED. It is ~240 MB of AdamW moments per checkpoint and the
#     tailnet link runs at ~0.6 MB/s. It is only needed to RESUME, which happens on the rig
#     where the file already lives.
#  2. ORDER. The adapter is ~130 MB (PEFT saves LoRA weights in fp32) and takes ~3.5 min to
#     cross. rsync walks alphabetically, so it would fetch `adapters/best/` before
#     `adapters/last/`. `last` is the run's actual final policy and the one artifact the
#     Definition of Done cannot do without, so it is pulled in its own pass, first.
set -u
RIG="shreyv@100.127.102.71"
KEY="$HOME/.ssh/id_ed25519"
DEST="$HOME/archive/cost-aware-vlm"
SSH="ssh -i $KEY -o ConnectTimeout=10 -o BatchMode=yes"
COMMON=(-az --partial --timeout=120 --exclude '*.tmp' --exclude '*.tmp/' --exclude 'trainer_state.pt')
mkdir -p "$DEST"
while true; do
  # Pass 1 — metrics, rollouts, crops, eval, config. Small and decisive.
  rsync "${COMMON[@]}" --exclude 'adapters/' -e "$SSH" \
    "$RIG:/srv/ai/runs/" "$DEST/" >> "$DEST/mirror.log" 2>&1
  # Pass 2 — the final adapter of each run, before anything else under adapters/.
  rsync "${COMMON[@]}" --include '*/' --include 'adapters/last/***' --exclude 'adapters/*' \
    -e "$SSH" "$RIG:/srv/ai/runs/" "$DEST/" >> "$DEST/mirror.log" 2>&1
  # Pass 3 — everything else, best/ included.
  rsync "${COMMON[@]}" -e "$SSH" \
    "$RIG:/srv/ai/runs/" "$DEST/" >> "$DEST/mirror.log" 2>&1
  echo "$(date '+%F %T') mirror cycle rc=$?" >> "$DEST/mirror.log"
  sleep 45
done
