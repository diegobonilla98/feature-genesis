import json
import time
from pathlib import Path

from telegram_training_helper import TelegramBot


PIPELINE_PID = 1590343
POLL_SECONDS = 60
PROJECT_ROOT = Path("/home/boni/projects/feature_genesis")
RUN_ROOT = PROJECT_ROOT / "runs/quickdraw_resnet34_gn_seed1729_v4"
ATTRIBUTION_ROOT = RUN_ROOT / "attribution/topk/v6_topk/continual/seed_0000"
INTERVENTION_ROOT = RUN_ROOT / "evaluation/step_00021975/topk/v6_topk/continual/seed_0000/interventions"
VALIDATION_PATH = RUN_ROOT / "artifact_validation.json"
ACCEPTED_FEATURES = [64, 65, 89, 99, 119, 146, 164, 178, 234, 282, 338, 387, 391, 392]


watcher_started = time.time()
while Path(f"/proc/{PIPELINE_PID}").exists():
    time.sleep(POLL_SECONDS)

time.sleep(5)

causal_count = sum(
    len(list((ATTRIBUTION_ROOT / f"feature_{feature_id:05d}").glob("counterfactual_*.json")))
    for feature_id in ACCEPTED_FEATURES
)
intervention_count = sum(
    len(list(INTERVENTION_ROOT.glob(f"feature_{feature_id:05d}_*.json")))
    for feature_id in ACCEPTED_FEATURES
)
validation = json.loads(VALIDATION_PATH.read_text()) if VALIDATION_PATH.exists() else {}
validation_is_fresh = VALIDATION_PATH.exists() and VALIDATION_PATH.stat().st_mtime >= watcher_started
passed = (
    validation_is_fresh
    and bool(validation.get("passed"))
    and causal_count == 28
    and intervention_count == 28
)
status = "PASSED" if passed else "NEEDS ATTENTION"
message = (
    "Feature Genesis causal pipeline finished.\n\n"
    f"Final artifact gate: {status}\n"
    f"Exact causal replays: {causal_count}/28\n"
    f"Removal and steering interventions: {intervention_count}/28\n"
    f"Report: {RUN_ROOT / 'reports/features/index.html'}"
)

bot = TelegramBot(auto_start=False, working_directory=str(PROJECT_ROOT))
bot.send_message(message)
