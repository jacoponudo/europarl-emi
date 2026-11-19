#!/bin/sh
# Six EMI methodologies, one after the other: 3 models (minilm = paper baseline, bge, gte) x 2 windows (150, 300).
# Each writes data/scores/<model>_<window>_<date>/ and its log in results/logs/scores/.
# Resumable: run it again with the same date and finished chunks are reused.
#
#   sh code/scores/run_variants.sh            # date 20261119 (the folder of the current baseline)
#   sh code/scores/run_variants.sh 20261120   # all six in new folders
#   tail -f results/logs/scores/run_variants.log
#
# With 300-word chunks (~400 tokens) the token limit is raised to 512, otherwise MiniLM (256) cuts the chunk end.
# mpnet (config.MODELS) is ready but not in the queue: add "run mpnet 150" / "run mpnet 300 ..." below.

set -u
cd "$(dirname "$0")/../.." || exit 1
DATE=${1:-20261119}
LOGS=results/logs/scores
mkdir -p "$LOGS"

run() {
    model=$1; window=$2; shift 2
    name=${model}_${window}_${DATE}
    echo "$(date '+%F %T') start $name" | tee -a "$LOGS/run_variants.log"
    if uv run python code/scores/score_unified.py --model "$model" --window "$window" --date "$DATE" "$@" \
        > "$LOGS/$name.log" 2>&1; then
        echo "$(date '+%F %T') done  $name: $(tail -2 "$LOGS/$name.log" | head -1)" | tee -a "$LOGS/run_variants.log"
    else
        echo "$(date '+%F %T') FAILED $name, see $LOGS/$name.log" | tee -a "$LOGS/run_variants.log"
    fi
}

run minilm 150
run minilm 300 --max-seq-length 512 --batch-size 64
run bge 150
run gte 150
run bge 300 --max-seq-length 512 --batch-size 64
run gte 300 --max-seq-length 512 --batch-size 64
echo "$(date '+%F %T') all finished" | tee -a "$LOGS/run_variants.log"
