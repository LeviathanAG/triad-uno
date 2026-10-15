"""Learned action-ranking policy with standard-library inference."""
import json
import math
from pathlib import Path

try:
    from learned_policy import FEATURE_NAMES, Policy
    from certified_skip_finish import skip_finish_plans
except ModuleNotFoundError:
    # SDK unit tests load this file directly from the repository root.
    from src.learned_policy import FEATURE_NAMES, Policy
    from src.certified_skip_finish import skip_finish_plans

_path = Path(__file__).resolve().parents[1] / "models" / "policy.json"
_model = json.loads(_path.read_text())
if (_model.get("format") != "triad-uno-route-linear-v1" or _model.get("features") != FEATURE_NAMES
        or not all(math.isfinite(w) for w in _model["weights"])):
    raise ValueError("invalid learned policy")
_policy = Policy(_model["weights"])


def initialize_state(game_id, seat, config):
    # Public organizer configuration, echoed through the permitted checkpoint.
    limit = config.get("max_turns") if isinstance(config, dict) else None
    if type(limit) is int and limit > 0:
        return {"uno_certified_finish_v1": {"max_turns": limit}}
    return None


def select_action(observation, agent_state):
    legal = observation["legal_actions"]
    if len(legal) == 1:
        return legal[0]["action_id"], agent_state
    if "hand" not in observation or "seat" not in observation:
        # Compatibility with the SDK's minimal hook fixtures.
        return next((a["action_id"] for a in legal if a["type"] == "play"), legal[0]["action_id"]), agent_state
    base = _policy.select(observation)
    saved = agent_state.get("uno_certified_finish_v1") if isinstance(agent_state, dict) else None
    limit = saved.get("max_turns") if isinstance(saved, dict) else None
    completed = observation.get("turns_completed")
    if type(limit) is int and limit > 0 and type(completed) is int and 0 <= completed < limit:
        plans = skip_finish_plans(observation, limit - completed)
        if plans and base not in plans:
            return next(a["action_id"] for a in legal if a["action_id"] in plans), agent_state
    return base, agent_state
