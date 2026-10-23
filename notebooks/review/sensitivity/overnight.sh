#!/usr/bin/env bash
# REVIEW ONLY. Runs the two sensitivity variants one after the other, then the comparison.
# Each step writes its own log; if one step fails, the next one still runs.
# Launch from the project folder (all speeches):
#   nohup caffeinate -i bash notebooks/review/sensitivity/overnight.sh > /dev/null 2>&1 &
# or on a fraction of speeches per year, e.g. 25%:
#   nohup caffeinate -i bash notebooks/review/sensitivity/overnight.sh 0.25 > /dev/null 2>&1 &
# Follow progress:  tail -f data/review/sensitivity/logs/overnight.log
# If it stops before the end, launch it again: finished parts are not recomputed.

cd "$(dirname "$0")/../../.." || exit 1
LOGS=data/review/sensitivity/logs
mkdir -p "$LOGS"
HERE=notebooks/review/sensitivity
EXTRA=()
[ -n "${1:-}" ] && EXTRA=(--sample "$1")

step() {
    name=$1; shift
    echo "$(date '+%F %T')  start  $name" >> "$LOGS/overnight.log"
    if "$@" > "$LOGS/$name.log" 2>&1; then
        echo "$(date '+%F %T')  done   $name" >> "$LOGS/overnight.log"
    else
        echo "$(date '+%F %T')  FAILED $name  (see $LOGS/$name.log)" >> "$LOGS/overnight.log"
    fi
}

step v1_mpnet    uv run python "$HERE/v1_mpnet.py" "${EXTRA[@]}"
step v2_word2vec uv run --with gensim python "$HERE/v2_word2vec.py" "${EXTRA[@]}"
step compare     uv run python "$HERE/compare.py"
echo "$(date '+%F %T')  all steps finished" >> "$LOGS/overnight.log"
