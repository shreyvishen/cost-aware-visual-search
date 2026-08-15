#!/usr/bin/env bash
# Wait for a run to write DONE on the rig, then pull its bundle at high priority and verify.
# This is the active enforcement of GOAL §11's "verify arrival before starting the next run".
set -u
RUN=${1:?usage: secure_run.sh run_a}
KEY="$HOME/.ssh/id_ed25519"
RIG="shreyv@100.127.102.71"
DEST="$HOME/archive/cost-aware-vlm"

until ssh -i "$KEY" -o ConnectTimeout=10 -o BatchMode=yes "$RIG" "test -f /srv/ai/runs/$RUN/DONE" 2>/dev/null; do
  sleep 20
done
echo "$(date '+%F %T') $RUN DONE on rig — pulling bundle"

# Small, decisive artifacts first; then the adapters.
for pass in "--exclude adapters/" ""; do
  rsync -az --partial --timeout=120 -e "ssh -i $KEY -o ConnectTimeout=10 -o BatchMode=yes" \
    --exclude 'trainer_state.pt' --exclude '*.tmp' --exclude '*.tmp/' $pass \
    "$RIG:/srv/ai/runs/$RUN/" "$DEST/$RUN/"
  echo "$(date '+%F %T') pass '${pass:-adapters}' rc=$?"
done

cd "$(dirname "$0")/.."
python3 scripts/verify_bundle.py --runs "$RUN"
echo "$(date '+%F %T') secure_run.sh $RUN finished with rc=$?"
