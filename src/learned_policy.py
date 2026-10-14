"""Own-hand sequence features for a frozen learned UNO base policy.

The bounded search imagines subsequent own plays with no intervening top-card
changes. It is a hand-structure feature, not a claim about future opponents.
Only filtered observations are read. No simulator or hidden state is imported.
"""
from collections import Counter
from functools import lru_cache
import math
try:
    from learned_base_policy import action_features as base_features, FEATURE_NAMES as BASE_NAMES
    from learned_beliefs import public_beliefs, symbol
except ModuleNotFoundError:
    from .learned_base_policy import action_features as base_features, FEATURE_NAMES as BASE_NAMES
    from .learned_beliefs import public_beliefs, symbol

EXTRA_NAMES = ["sequence_fraction", "sequence_stranded", "ending_wild",
               "next_colour_absence", "next_symbol_absence", "next_colour_preference"]
FEATURE_NAMES = BASE_NAMES + EXTRA_NAMES


def route_length(hand, colour, top, max_states=128, max_depth=8):
    """Longest discovered normal-play path, with a fixed deterministic budget."""
    cards = tuple(sorted(Counter((c.get('color') or '', c['kind'], c.get('value') if c.get('value') is not None else -1)
                                 for c in hand).items()))
    types = tuple(c for c, _ in cards)
    counts = tuple(n for _, n in cards)
    visits = 0

    @lru_cache(maxsize=256)
    def walk(counts, active, last_kind, last_value, depth):
        nonlocal visits
        if not depth or visits >= max_states:
            return 0
        visits += 1
        best = 0
        active_count = sum(n for n, (c, _, _) in zip(counts, types) if c == active)
        for i, ((c, kind, value), n) in enumerate(zip(types, counts)):
            if not n:
                continue
            wild = kind in ('wild', 'wild_draw_four')
            legal = (kind == 'wild' or (kind == 'wild_draw_four' and not active_count)
                     or c == active or (kind == 'number' and last_kind == 'number' and value == last_value)
                     or kind in ('skip', 'reverse', 'draw_two') and kind == last_kind)
            if not legal:
                continue
            reduced = counts[:i] + (n - 1,) + counts[i + 1:]
            options = sorted({other[0] for other, number in zip(types, reduced) if number and other[0]}) if wild else [c]
            if not options:
                options = [active]
            for chosen in options:
                length = 1 + walk(reduced, chosen, kind, value, depth - 1)
                best = max(best, length)
                if best >= min(depth, sum(counts)):
                    return best
        return best
    return walk(counts, colour or '', top.get('kind'), top.get('value', -1), min(max_depth, len(hand)))


def action_features(observation, action, beliefs=None, cache=None):
    beliefs = beliefs or public_beliefs(observation)
    play = action['type'] == 'play'
    card = action.get('card') or next((c for c in observation['hand'] if c['card_ref'] == action.get('card_ref')), {})
    colour = action.get('color') or card.get('color') or observation.get('active_color')
    remaining = [c for c in observation['hand'] if not(play and c['card_ref'] == action.get('card_ref'))]
    top = card if play else observation.get('top_discard', {})
    key = (tuple(sorted((c.get('color') or '', c['kind'], c.get('value', -1) if c.get('value') is not None else -1)
                        for c in remaining)), colour, top.get('kind'), top.get('value'))
    if cache is not None and key in cache:
        length = cache[key]
    else:
        length = route_length(remaining, colour, top)
        if cache is not None:
            cache[key] = length
    sizes = observation['hand_sizes']
    direction = observation.get('direction', 1)
    if play and card.get('kind') == 'reverse':
        direction = -direction
    active = [s for s in range(len(sizes)) if s not in observation.get('finished_order', [])]
    successor = observation['seat']
    steps = 2 if play and card.get('kind') in ('skip', 'draw_two', 'wild_draw_four') else 1
    for _ in range(steps):
        successor = next((successor + distance * direction) % len(sizes) for distance in range(1, len(sizes) + 1)
                         if (successor + distance * direction) % len(sizes) in active)
    denom = min(8, len(remaining))
    extras = [length / max(1, denom), (denom - length) / 8.,
              float(len(remaining) == 1 and remaining[0]['kind'] in ('wild', 'wild_draw_four')),
              beliefs['colours'].get((successor, colour), 0.),
              beliefs['symbols'].get((successor, symbol(top)), 0.),
              float(beliefs['preference'].get(successor) == colour)]
    return base_features(observation, action) + extras


class Policy:
    def __init__(self, weights):
        if len(weights) != len(FEATURE_NAMES) or not all(math.isfinite(w) for w in weights):
            raise ValueError('invalid route policy weights')
        self.weights = list(weights)

    def select(self, observation):
        legal = observation['legal_actions']
        if len(legal) == 1:
            return legal[0]['action_id']
        beliefs, cache = public_beliefs(observation), {}
        return max(legal, key=lambda a: sum(w * x for w, x in zip(self.weights, action_features(observation, a, beliefs, cache))))['action_id']
