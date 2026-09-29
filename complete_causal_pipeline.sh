#!/usr/bin/env bash
set -euo pipefail

cd /home/boni/projects/feature_genesis
source /home/boni/ai/envs/dgx-dl/bin/activate

while pgrep -f '^python scripts/08_rank_training_batches.py$' >/dev/null
do
    sleep 30
done

python scripts/09_counterfactual_replay.py
python scripts/18_summarize_causal_validation.py
python scripts/17_render_training_influence.py
python scripts/12_evaluate_intervention.py
python scripts/19_build_experiment_summary.py
python scripts/21_generate_feature_dossiers.py
python scripts/10_make_report.py
python scripts/20_validate_experiment_artifacts.py
