"""Original compact UNO response priors from permitted own/public information.

Uniform sampling without replacement is an explicit approximation: opposing
policies, soft colour preferences and joint constraints across seats are not
modelled. These probabilities are features, not factual opponent hands.
No trusted simulator state or real shuffle seed is accepted.
"""
from uno_public_belief import public_unknown_pool, proven_constraints, compatible
from uno_policy_v2 import public_beliefs

FEATURE_NAMES = [
    'unseen_responder_can_play', 'unseen_responder_one_card_finish',
    'unseen_attack_stack', 'unseen_attack_one_card_finish',
    'unseen_after_accept_one_card_finish', 'unseen_penalty_draw_pressure',
]


def any_hit_probability(total, hits, hand_size):
    """Hypergeometric P(at least one hit), with finite-card validation."""
    if not 0 <= hits <= total or not 0 <= hand_size <= total:
        raise ValueError('impossible finite-card counts')
    if not hand_size or not hits:
        return 0.0
    if hand_size > total - hits:
        return 1.0
    none = 1.0
    for offset in range(hand_size):
        none *= (total - hits - offset) / (total - offset)
    return 1.0 - none


def responds_normally(card_key, active_color, top):
    color, kind, value = card_key
    if kind in ('wild', 'wild_draw_four'):
        # A no-response hand has no active-colour cards, so WD4 is legal
        # whenever it is the only possible response. This makes the union
        # test exact for the no-legal-action event under this uniform prior.
        return True
    if color == active_color:
        return True
    if kind == 'number' and top.get('kind') == 'number':
        return value == top.get('value')
    return kind in ('skip', 'reverse', 'draw_two') and kind == top.get('kind')


class PreparedRisks:
    """Prepare one observation once; reuse across its legal candidates."""
    def __init__(self, observation):
        self.observation = observation
        self.pool, _ = public_unknown_pool(observation)
        self.constraints = proven_constraints(observation)
        self.pending_penalty = public_beliefs(observation)['penalty']
        self.hand = {card['card_ref']: card for card in observation['hand']}
        self.seat = observation['seat']
        self.sizes = observation['hand_sizes']
        finished = set(observation.get('finished_order', []))
        self.active = [seat for seat in range(len(self.sizes)) if seat not in finished]
        self.seat_pools = {}
        for seat in self.active:
            if seat != self.seat:
                self.seat_pools[seat] = [
                    (key, count) for key, count in self.pool.items()
                    if compatible(key, self.constraints[seat])]

    def next_seat(self, seat, direction):
        return next((seat + distance * direction) % len(self.sizes)
                    for distance in range(1, len(self.sizes) + 1)
                    if (seat + distance * direction) % len(self.sizes) in self.active)

    def probability(self, seat, predicate):
        if seat == self.seat:
            return 0.0
        cards = self.seat_pools[seat]
        total = sum(count for _, count in cards)
        hits = sum(count for key, count in cards if predicate(key))
        return any_hit_probability(total, hits, self.sizes[seat])

    def features(self, action):
        observation = self.observation
        play = action['type'] == 'play'
        card = self.hand.get(action.get('card_ref'), {})
        if play and len(self.hand) == 1:
            return [0.0] * len(FEATURE_NAMES)
        if action['type'] == 'choose_color':
            return [0.0] * len(FEATURE_NAMES)
        color = action.get('color') or card.get('color') or observation.get('active_color')
        top = card if play else observation['top_discard']
        direction = observation.get('direction', 1)
        kind = card.get('kind') if play else None
        if kind == 'reverse':
            direction = -direction
        next_seat = self.next_seat(self.seat, direction)
        responder = self.next_seat(next_seat, direction) if kind == 'skip' else next_seat
        ordinary = lambda key: responds_normally(key, color, top)
        if kind in ('draw_two', 'wild_draw_four'):
            stack = lambda key: key[1] == 'wild_draw_four' or (kind == 'draw_two' and key[1] == 'draw_two')
            response = self.probability(next_seat, stack)
            stack_finish = response if self.sizes[next_seat] == 1 else 0.0
            after = self.next_seat(next_seat, direction)
            after_finish = self.probability(after, ordinary) if after != self.seat and self.sizes[after] == 1 else 0.0
            pressure = (1.0 - response) * min(24, self.pending_penalty + (2 if kind == 'draw_two' else 4)) / 24.0
            return [response, stack_finish, response, stack_finish,
                    (1.0 - response) * after_finish, pressure]
        response = self.probability(responder, ordinary)
        finish = response if responder != self.seat and self.sizes[responder] == 1 else 0.0
        return [response, finish, 0.0, 0.0, 0.0, 0.0]
