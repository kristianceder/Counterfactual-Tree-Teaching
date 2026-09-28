#!/bin/bash
# Reproduce every run under results/runs (the paper's experiments).
#
#     scripts/run_experiments.sh baselines   # no recovery, scripted rules, GPT-6 Luna alone
#     scripts/run_experiments.sh arms        # method / distillation alone / experience without replays, x3,
#                                            #   each frozen at every 25th episode on the held-out layouts
#                                            #   and at episode 200 on the unseen layouts
#     scripts/run_experiments.sh permap      # per-map self-improvement: 8 layouts x 6 episodes x 3, then frozen
#     scripts/run_experiments.sh all
#
#     OUT=/some/dir scripts/run_experiments.sh arms    # write somewhere other than results/runs
#     DRY=1 scripts/run_experiments.sh all             # print the commands, run nothing
#
# A run whose recording is already complete is skipped, so an interrupted queue can be restarted.
# The runs with `--teacher llm` need OPENAI_API_KEY; the rest need no key. Layouts: seeds 100-299 are
# the learning stream, 300-319 the held-out layouts, 400-499 the unseen ones.
#
# Wall time on a laptop CPU: a 200-episode learning run is 0.5 h without replays and 1.5-2 h with
# them, a frozen 20-episode evaluation about 4 min, the 100 unseen layouts about 20 min. The solver is
# capped by wall time (500 ms), so results depend somewhat on the machine and its load: keep it on
# mains power and otherwise lightly used, and expect the same picture, not the same step numbers.
set -u
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python}"
D="${OUT:-results/runs}"
mkdir -p "$D"
export MPLBACKEND=Agg

complete() {  # complete NAME EPISODES: the recording holds that many episodes
  [ -f "$D/$1/run.json" ] && [ "$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['episodes']))" "$D/$1/run.json")" = "$2" ]
}
run() {  # run NAME EPISODES ARGS...: one scripts/run.py invocation, logged to $D/NAME.log
  local name=$1 episodes=$2; shift 2
  if complete "$name" "$episodes"; then echo "skip  $name (done)"; return; fi
  if [ "${DRY:-0}" = 1 ]; then echo "$PY scripts/run.py --episodes $episodes $* --record $D/$name"; return; fi
  echo "$(date +%H:%M:%S) start $name"
  "$PY" scripts/run.py --episodes "$episodes" "$@" --record "$D/$name" > "$D/$name.log" 2>&1
  echo "$(date +%H:%M:%S) done  $name (exit $?)"
}
frozen() {  # frozen NAME TREE EPISODES SEED: a stored tree, no teacher, nothing learned or retired
  run "$1" "$3" --teacher none --seed "$4" --no-learn --no-save --tree "$2"
}

baselines() {
  run no_recovery_heldout 20 --teacher none --seed 300 --reset --no-learn --no-save --tree "$D/no_recovery_heldout.json"
  run scripted_heldout 20 --teacher scripted --seed 300 --reset --no-learn --no-save --tree "$D/scripted_heldout.json"
  run scripted_learning 200 --teacher scripted --seed 100 --reset --no-learn --no-save --tree "$D/scripted_learning.json"
  run scripted_unseen 100 --teacher scripted --seed 400 --reset --no-learn --no-save --tree "$D/scripted_unseen.json"
  for r in 1 2 3; do   # the teacher alone: an empty tree that learns nothing, so every tick is a call
    run "luna_heldout_r$r" 20 --teacher llm --seed 300 --reset --no-learn --no-save --tree "$D/luna_heldout_r$r.json"
  done
}

arm() {  # arm NAME FLAGS...: learn on the 200-layout stream, then evaluate its checkpoints frozen
  local name=$1; shift
  run "$name" 200 --teacher llm --seed 100 --reset --checkpoint-every 25 --tree "$D/$name.json" "$@"
  for c in 0025 0050 0075 0100 0125 0150 0175 0200; do
    [ -f "$D/$name.ep$c.json" ] || [ "${DRY:-0}" = 1 ] || continue
    frozen "${name}_heldout_ep$c" "$D/$name.ep$c.json" 20 300
  done
  [ -f "$D/$name.ep0200.json" ] || [ "${DRY:-0}" = 1 ] || return
  frozen "${name}_unseen" "$D/$name.ep0200.json" 100 400
}

arms() {
  for r in 1 2 3; do
    arm "method_r$r" --memory both --simulate --cf-retire-after 10       # experience, with replays that retire
    arm "distillation_r$r" --memory off --simulate                      # replays only label, nothing shown
    arm "experience_noreplay_r$r" --memory both                         # experience without replays
  done
}

permap() {
  for r in 1 2 3; do
    for s in 300 302 304 305 307 315 319 461; do
      local name="permap_${s}_r$r"
      run "$name" 6 --teacher llm --seed "$s" --no-vary-seed --reset --memory both --simulate --cf-retire-after 10 \
          --tree "$D/$name.json"
      [ -f "$D/$name.json" ] || [ "${DRY:-0}" = 1 ] || continue
      run "${name}_frozen" 1 --teacher none --seed "$s" --no-vary-seed --no-learn --no-save --tree "$D/$name.json"
    done
  done
}

case "${1:-}" in
  baselines) baselines ;;
  arms) arms ;;
  permap) permap ;;
  all) baselines; arms; permap ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
