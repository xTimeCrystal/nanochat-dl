#!/bin/bash
# Queue a latent_think.sh run behind a running one: wait for process <pid> (the running torchrun) to exit, then launch.
#   setsid -f bash runs/latent_think_after.sh <pid> <run-name> [base_train args...]
pid=$1; shift
[ "$pid" = 0 ] || while kill -0 $pid 2>/dev/null; do sleep 20; done # (0: start at once)
exec bash "$(dirname "$0")/latent_think.sh" "$@"
