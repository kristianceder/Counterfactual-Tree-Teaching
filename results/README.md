# Recorded runs

Every run behind the paper's tables and figures, as written by `scripts/run.py` (commands in
`scripts/run_experiments.sh`). Layout seeds: 100-299 learning stream, 300-319 held-out, 400-499
unseen. All runs use the scenario grid's mixed stream (option 1) and GPT-6 Luna as the teacher where
there is one.

| run | what it is |
|---|---|
| `no_recovery_heldout` | the MPC alone (every tick answered "carry on"), held-out layouts |
| `scripted_heldout`, `scripted_learning`, `scripted_unseen` | the scripted rules on the held-out, learning and unseen layouts; `scripted_learning/` also holds the replay labels of its recoveries (below) |
| `luna_heldout_r{1,2,3}` | GPT-6 Luna alone (empty tree, nothing learned), held-out layouts |
| `method_r{1,2,3}` | the method: tree learned from the teacher, experience shown to it, replay labels that retire rules |
| `distillation_r{1,2,3}` | ablation: the tree learned from the teacher alone (replays run and label, nothing shown or retired) |
| `experience_noreplay_r{1,2,3}` | ablation: experience shown to the teacher, no replays |
| `<arm>_r{n}_heldout_epNNNN` | that arm's tree after NNNN learning episodes, frozen, on the held-out layouts |
| `<arm>_r{n}_unseen` | its final tree, frozen, on the 100 unseen layouts |
| `permap_<seed>_r{1,2,3}` | one layout six times in a row, tree grown from empty in the method's configuration |
| `permap_<seed>_r{n}_frozen` | that tree frozen, one more episode on the same layout |

Files per run `NAME`:

- `NAME/` -- the recording: `run.json` (run settings and a per-episode summary) and
  `episode_NNN.json` per episode: the layout (`meta.layout`: staged situations and whether a human is
  needed), the robot and obstacles per control step (`frames`), every decision with the step of the
  report it was drawn from, the step it was acted on and who made it (`decisions`: tree, model or the
  recovery-loop breaker), the recoveries executed, the tree's edits and the judge's verdicts
  (`judgements`), and the summary (ending, steps, agreement with the scripted rules, replay labels).
- `NAME.json` -- the tree at the end of the run, with its edit log; `NAME.epNNNN.json` its checkpoints.
- `NAME.experience.json` -- the experience table (judge verdicts and replay labels per situation and action).
- `NAME.jsonl` -- every teacher call: the report it was shown, the experience text, and its answer.
- `NAME.api.jsonl` -- every API call: latency, token usage (with cache reads) and cost.
- `NAME.log` -- the run's console output.
- `scripted_learning/counterfactual_epNNN.json` -- the scripted rules run no replays of their own, so
  their learning stream was replayed after the fact, with the settings the method's loop uses (grace
  50, slack 10), by the script `scripts/replay_run.py` was ported from (same labels, same file
  format): the exactness check (`check`) and, per recovery, the
  replay without it and its label (`counterfactuals`). Of 200 episodes 160 replayed exactly; 329 of
  351 recoveries were clean to their own step. The `run_dir` inside is where the run lived when it was
  replayed.

The `tree` path inside a recording's `run.json` is where the tree lived when the run was made.
