#!/usr/bin/env bash
# Watchdog (GOAL §20): a run that hangs past its timer plus a grace period must not burn the
# night silently. We SIGTERM it — train.py installs a handler that checkpoints and finalises —
# rather than kill -9, so the artifacts survive.
set -u
RUN=${1:?usage: watchdog.sh run_a <deadline_epoch>}
DEADLINE=${2:?}
KEY="$HOME/.ssh/id_ed25519"; RIG="shreyv@100.127.102.71"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if ssh -i "$KEY" -o ConnectTimeout=10 -o BatchMode=yes "$RIG" "test -f /srv/ai/runs/$RUN/DONE" 2>/dev/null; then
    echo "$(date '+%F %T') $RUN finished before its deadline; watchdog standing down"
    exit 0
  fi
  sleep 60
done
echo "$(date '+%F %T') WATCHDOG: $RUN passed its deadline without DONE. Sending SIGTERM so it checkpoints."
ssh -i "$KEY" -o ConnectTimeout=10 "$RIG" "pkill -TERM -f 'src.train --config configs/$RUN.json'"
echo "$(date '+%F %T') SIGTERM sent to $RUN"
