#!/usr/bin/env bash
# Final wrap. Everything here reads MAC-LOCAL artifacts only — it must work with the rig off.
# Safe to run repeatedly; it writes results.md and out/demo.html and nothing else.
set -u
cd "$(dirname "$0")/.."

echo "=================================================================="
echo " WRAP  $(date '+%F %H:%M:%S %Z')"
echo "=================================================================="

echo
echo "---- 1. Definition of Done ------------------------------------------------"
python3 scripts/verify_bundle.py
DOD=$?

echo
echo "---- 2. Results table -----------------------------------------------------"
python3 scripts/compare_runs.py > /dev/null 2>&1
sed -n '1,80p' results.md 2>/dev/null

echo
echo "---- 3. Demo page ---------------------------------------------------------"
if python3 scripts/build_demo.py > /tmp/build_demo.log 2>&1; then
  ls -la out/demo.html 2>/dev/null && echo "demo rebuilt"
else
  echo "demo build FAILED — see /tmp/build_demo.log"; tail -5 /tmp/build_demo.log
fi

echo
echo "---- 4. Artifact sizes ----------------------------------------------------"
du -sh "$HOME/archive/cost-aware-vlm"/* 2>/dev/null | sort -k2

echo
echo "---- 5. Mirror health -----------------------------------------------------"
tail -3 "$HOME/archive/cost-aware-vlm/mirror.log" 2>/dev/null
pgrep -f "scripts/mirror.sh" > /dev/null && echo "mirror: running" || echo "mirror: NOT RUNNING"

echo
echo "=================================================================="
if [ $DOD -eq 0 ]; then
  echo " DEFINITION OF DONE: MET — A and B are on this Mac and comparable."
else
  echo " DEFINITION OF DONE: NOT MET — see section 1."
fi
echo "=================================================================="
