"""Learned action-ranking policy with standard-library inference."""
import json
import math
from pathlib import Path

try:
    from learned_policy import FEATURE_NAMES, Policy
except ModuleNotFoundError:
    # SDK unit tests load this file directly from the repository root.
    from src.learned_policy import FEATURE_NAMES, Policy

_path = Path(__file__).resolve().parents[1] / "models" / "policy.json"
_model = json.loads(_path.read_text())
if (_model.get("format") != "triad-uno-route-linear-v1" or _model.get("features") != FEATURE_NAMES
        or not all(math.isfinite(w) for w in _model["weights"])):
    raise ValueError("invalid learned policy")
_policy = Policy(_model["weights"])


def select_action(observation, agent_state):
    legal = observation["legal_actions"]
    if len(legal) == 1:
        return legal[0]["action_id"], agent_state
    if "hand" not in observation or "seat" not in observation:
        # Compatibility with the SDK's minimal hook fixtures.
        return next((a["action_id"] for a in legal if a["type"] == "play"), legal[0]["action_id"]), agent_state
    return _policy.select(observation), agent_state
