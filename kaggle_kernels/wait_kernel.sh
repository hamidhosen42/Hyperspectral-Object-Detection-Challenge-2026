#!/bin/zsh
# usage: kaggle_kernels/wait_kernel.sh <slug> [--new]
# Polls a kernel until it finishes (COMPLETE/ERROR/CANCEL), then downloads its small outputs (csv/json/log/.pt) to kaggle_out/<slug>.
# --new: after a fresh push, first wait until the new version is QUEUED/RUNNING (the old version's status shows until then).
cd "$(dirname "$0")/.."; export PATH="$PWD/.venv/bin:$PATH"; S=$1
if [ "$2" = "--new" ]; then for i in $(seq 1 30); do case "$(kaggle kernels status hosen42/$S 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | cut -d. -f2)" in QUEUED|RUNNING) break;; esac; sleep 30; done; fi
while true; do st=$(kaggle kernels status hosen42/$S 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | cut -d. -f2); case "$st" in COMPLETE|ERROR|CANCEL*) break;; esac; sleep 120; done
mkdir -p kaggle_out/$S
for pat in '\.csv$' '\.json$' '\.log$' 'best\.pt$'; do kaggle kernels output hosen42/$S -p kaggle_out/$S --file-pattern "$pat" -o >/dev/null 2>&1; done
echo "$S finished: $st at $(date -u +%H:%M) UTC"; ls -R kaggle_out/$S | head -20
