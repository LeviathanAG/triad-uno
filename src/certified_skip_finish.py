"""Original own/public detector for uninterrupted two-player Skip finishes.

Plans are conditional on enough placement turns before the global turn limit.
References describe the current hand only; recompute after each observation.
No opponent cards, simulator state, random seed or learned evaluator is used.
This isolated development helper is not part of the submitted runtime.
"""


def skip_finish_plans(observation, remaining_placement_turns=None):
    """Return legal initial actions whose Skip chain finishes without a rival turn.

    Unknown turn-limit headroom is deliberately insufficient for a proof.
    Every nonfinal placement must be a Skip with exactly two unfinished seats.
    """
    hand = observation.get('hand', [])
    if not hand or len(hand) > 12 or remaining_placement_turns is None:
        return {}
    if remaining_placement_turns < len(hand):
        return {}
    sizes = observation.get('hand_sizes', [])
    finished = set(observation.get('finished_order', []))
    active = [seat for seat in range(len(sizes)) if seat not in finished]
    if len(active) != 2 or observation.get('seat') not in active:
        return {}
    if observation.get('phase') not in ('TURN', 'AFTER_DRAW'):
        return {}
    by_ref = {card['card_ref']: card for card in hand}
    plans = {}
    for action in observation.get('legal_actions', []):
        first = by_ref.get(action.get('card_ref'))
        if action.get('type') != 'play' or not first or first['kind'] != 'skip':
            continue
        others = [card for card in hand if card['card_ref'] != first['card_ref']]
        skips = [card for card in others if card['kind'] == 'skip']
        normal = [card for card in others if card['kind'] != 'skip']
        if len(normal) > 1:
            continue
        if normal and normal[0]['kind'] not in ('wild', 'wild_draw_four'):
            color = normal[0]['color']
            if skips:
                last = next((card for card in skips if card['color'] == color), None)
                if last is None:
                    continue
                skips = [card for card in skips if card is not last] + [last]
            elif first['color'] != color:
                continue
        plan = [first] + skips + normal
        plans[action['action_id']] = {
            'current_hand_refs': [card['card_ref'] for card in plan],
            'required_placement_turns': len(plan),
        }
    return plans
