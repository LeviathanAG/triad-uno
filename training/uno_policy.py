"""Linear action-ranking policy; inference uses filtered observations only."""
from collections import Counter
import json
from pathlib import Path

FEATURE_NAMES = [
    "play", "number", "skip", "reverse", "draw_two", "wild", "wild_draw_four",
    "remaining_colour_count", "remaining_colour_fraction", "remaining_match_count",
    "attack_next_threat", "reverse_escape_threat", "wild_when_small_hand",
    "finish", "keep", "choose_colour", "next_can_finish", "own_hand_size",
]
INITIAL_WEIGHTS = [8., .5, -.2, -.2, -.2, -2., -3., 2., 1., 1., 4., 2., 2., 12., -4., 0., 0., 0.]


def action_features(observation, action):
    hand = observation["hand"]
    by_ref = {card["card_ref"]: card for card in hand}
    card = action.get("card") or by_ref.get(action.get("card_ref")) or {}
    kind = card.get("kind")
    play = action["type"] == "play"
    colour = action.get("color") or card.get("color") or observation.get("active_color")
    remaining = [c for c in hand if not (play and c["card_ref"] == action.get("card_ref"))]
    colours = Counter(c.get("color") for c in remaining if c.get("color"))
    matching = sum(c.get("color") == colour or c.get("kind") in ("wild", "wild_draw_four")
                   or (kind == "number" and c.get("kind") == "number" and c.get("value") == card.get("value"))
                   or (kind not in (None, "number", "wild", "wild_draw_four") and c.get("kind") == kind)
                   for c in remaining)
    sizes = observation.get("hand_sizes", [len(hand)] * 4)
    seat, direction = observation["seat"], observation.get("direction", 1)
    active = [s for s in range(len(sizes)) if s not in observation.get("finished_order", [])]
    def neighbour(d):
        for distance in range(1, len(sizes)):
            other = (seat + distance * d) % len(sizes)
            if other in active:
                return max(1, sizes[other])
        return max(1, len(hand))
    next_size, previous_size = neighbour(direction), neighbour(-direction)
    threat = 1. / next_size
    attack = play and kind in ("skip", "draw_two", "wild_draw_four")
    return [float(play), float(kind == "number"), float(kind == "skip"),
            float(kind == "reverse"), float(kind == "draw_two"), float(kind == "wild"),
            float(kind == "wild_draw_four"), min(colours[colour], 12) / 4.,
            colours[colour] / max(1, len(remaining)), min(matching, 12) / 4.,
            float(attack) * threat,
            float(play and kind == "reverse") * (threat - 1. / previous_size),
            float(play and kind in ("wild", "wild_draw_four")) / max(1, len(hand)),
            float(play and len(hand) == 1), float(action["type"] == "keep"),
            float(action["type"] == "choose_color"), float(play) * float(next_size <= 2),
            float(play) * min(len(hand), 24) / 12.]


class Policy:
    def __init__(self, weights):
        if len(weights) != len(FEATURE_NAMES):
            raise ValueError("wrong policy feature count")
        self.weights = [float(w) for w in weights]

    def select(self, observation):
        legal = observation["legal_actions"]
        return max(legal, key=lambda action: sum(w * x for w, x in
                   zip(self.weights, action_features(observation, action))))["action_id"]

    def save(self, path, metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"format": "triad-uno-linear-v1",
            "features": FEATURE_NAMES, "weights": self.weights,
            "training": metadata}, indent=2) + "\n")
