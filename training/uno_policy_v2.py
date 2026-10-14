"""History-aware non-language UNO policy with public-information beliefs."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from uno_policy import FEATURE_NAMES as V1_FEATURES, action_features as v1_features

EXTRA_FEATURES = [
    "successor_colour_absence", "successor_symbol_absence", "successor_colour_preference",
    "successor_finish_risk", "remaining_colour_components", "largest_component_fraction",
    "remaining_bridge_symbols", "remaining_duplicates", "remaining_colour_entropy",
    "remaining_wilds", "chosen_colour_exhausted", "stack_penalty_play",
    "successor_one_card_colour_absence", "successor_one_card_symbol_absence",
]
FEATURE_NAMES = V1_FEATURES + EXTRA_FEATURES


def symbol(card):
    kind = card.get("kind")
    if kind == "number":
        return (kind, card.get("value"))
    return (kind, None) if kind in ("skip", "reverse", "draw_two") else None


def public_beliefs(observation):
    """Fold public events; a normal forced draw reveals pre-draw absences.

    Unknown draws soften older evidence. Auto-kept or played drawn cards allow
    exact post-draw absence deductions. Penalty draws do not imply any absence.
    No simulator state, private opponent cards or seed is accessed here.
    """
    lack_colour, lack_symbol, preference, pending_draw = {}, {}, {}, {}
    active_colour, top_symbol, pending_penalty = None, None, 0
    for event in observation.get("history", []):
        kind, seat = event.get("type"), event.get("seat")
        if kind in ("game_start", "initial_discard"):
            card = event.get("card", event.get("initial_discard", {})) or {}
            active_colour = card.get("color", active_colour)
            top_symbol = symbol(card)
        elif kind == "initial_color_choice":
            active_colour = event["color"]
            preference[seat] = active_colour
        elif kind in ("draw", "penalty_draw"):
            count = event.get("drawn", 0) if kind == "draw" else event.get("count", 0)
            for key in list(lack_colour):
                if key[0] == seat:
                    lack_colour[key] *= .75 ** count
            for key in list(lack_symbol):
                if key[0] == seat:
                    lack_symbol[key] *= .92 ** count
            if count:
                preference.pop(seat, None)
            if kind == "draw":
                if active_colour:
                    lack_colour[(seat, active_colour)] = .75 ** count
                if top_symbol:
                    lack_symbol[(seat, top_symbol)] = .92 ** count
                pending_draw[seat] = (event.get("turn_number"), active_colour, top_symbol)
            else:
                pending_penalty = 0
        elif kind in ("keep_auto", "play"):
            card = event.get("card") or {}
            if kind == "play":
                colour, played_symbol = card.get("color"), symbol(card)
                if colour:
                    lack_colour[(seat, colour)] = 0.
                if played_symbol:
                    lack_symbol[(seat, played_symbol)] = 0.
            drawn = pending_draw.get(seat)
            if drawn and drawn[0] == event.get("turn_number") and (kind == "keep_auto" or event.get("from_draw")):
                if drawn[1]:
                    lack_colour[(seat, drawn[1])] = 1.
                if drawn[2]:
                    lack_symbol[(seat, drawn[2])] = 1.
                pending_draw.pop(seat, None)
            if kind == "play":
                active_colour = event.get("active_color", card.get("color"))
                top_symbol = symbol(card)
                if card.get("kind") in ("wild", "wild_draw_four"):
                    preference[seat] = active_colour
                if "pending_penalty" in event:
                    pending_penalty = event["pending_penalty"]
    return {"colours": lack_colour, "symbols": lack_symbol, "preference": preference, "penalty": pending_penalty}


def structure(hand):
    counts = Counter(c.get("color") for c in hand if c.get("color"))
    groups = defaultdict(set)
    for card in hand:
        key = symbol(card)
        if key and card.get("color"):
            groups[key].add(card["color"])
    adjacency = {colour: {colour} for colour in counts}
    for colours in groups.values():
        for colour in colours:
            adjacency[colour].update(colours)
    components, visited = [], set()
    for colour in counts:
        if colour in visited:
            continue
        frontier, connected = [colour], set()
        while frontier:
            current = frontier.pop()
            if current not in connected:
                connected.add(current)
                frontier.extend(adjacency[current] - connected)
        visited.update(connected)
        components.append(sum(counts[c] for c in connected))
    wilds = sum(c.get("kind") in ("wild", "wild_draw_four") for c in hand)
    if wilds and components:
        components = [sum(components)]
    coloured = sum(counts.values())
    entropy = -sum((n / coloured) * math.log(n / coloured) for n in counts.values()) if coloured else 0.
    identical = Counter((c.get("color"), c.get("kind"), c.get("value")) for c in hand)
    return [len(components) / 4., max(components, default=0) / max(1, len(hand)),
            sum(len(colours) > 1 for colours in groups.values()) / 8.,
            sum(n - 1 for n in identical.values()) / 4., entropy / 2., wilds / 4.]


def action_features(observation, action, beliefs=None):
    beliefs = beliefs or public_beliefs(observation)
    hand = observation["hand"]
    play = action["type"] == "play"
    card = action.get("card") or next((c for c in hand if c["card_ref"] == action.get("card_ref")), {})
    kind = card.get("kind")
    colour = action.get("color") or card.get("color") or observation.get("active_color")
    remaining = [c for c in hand if not (play and c["card_ref"] == action.get("card_ref"))]
    sizes, seat = observation["hand_sizes"], observation["seat"]
    direction = observation.get("direction", 1)
    active = [s for s in range(len(sizes)) if s not in observation.get("finished_order", [])]
    if play and kind == "reverse":
        direction = -direction
    steps = 2 if play and kind in ("skip", "draw_two", "wild_draw_four") else 1
    successor = seat
    for _ in range(steps):
        for distance in range(1, len(sizes) + 1):
            other = (successor + distance * direction) % len(sizes)
            if other in active:
                successor = other
                break
    absence = beliefs["colours"].get((successor, colour), 0.)
    absent_symbol = beliefs["symbols"].get((successor, symbol(card)), 0.)
    threat = 1. / max(1, sizes[successor])
    one_card = float(sizes[successor] == 1)
    extras = [absence, absent_symbol, float(beliefs["preference"].get(successor) == colour),
              threat * (1. - absence), *structure(remaining),
              float(not any(c.get("color") == colour for c in remaining)),
              float(play) * min(beliefs["penalty"], 24) / 12.,
              one_card * absence, one_card * absent_symbol]
    return v1_features(observation, action) + extras


class Policy:
    def __init__(self, weights):
        if len(weights) != len(FEATURE_NAMES) or not all(math.isfinite(float(w)) for w in weights):
            raise ValueError("invalid v2 policy weights")
        self.weights = list(weights)

    def select(self, observation):
        legal = observation["legal_actions"]
        if len(legal) == 1:
            return legal[0]["action_id"]
        beliefs = public_beliefs(observation)
        return max(legal, key=lambda action: sum(w * x for w, x in
                   zip(self.weights, action_features(observation, action, beliefs))))["action_id"]

    def save(self, path, metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"format": "triad-uno-linear-v2", "features": FEATURE_NAMES,
            "weights": self.weights, "training": metadata}, indent=2) + "\n")
