#!/bin/bash
# A sequence of latent_think.sh runs behind a running one: wait for process <pid>, then each "<run-name> <args...>" in turn
# (a failed run does not stop the rest).
#   setsid -f bash runs/latent_think_queue.sh <pid> "d8-baseline-s1 --depth=8 --seed=1" "d8-fbt2-s1 --depth=8 --fbt-passes=2 --seed=1"
# (<pid>: a running run to wait for, or 0 to start at once)
pid=$1; shift
[ "$pid" = 0 ] || while kill -0 $pid 2>/dev/null; do sleep 20; done # (0: start at once)
for spec in "$@"; do
  echo "$(date -u +%H:%M) start $spec"
  bash "$(dirname "$0")/latent_think.sh" $spec
  echo "$(date -u +%H:%M) end $spec rc=$?"
done
