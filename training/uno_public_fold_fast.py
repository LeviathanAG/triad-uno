"""Development-only incremental public-event fold for append-only simulator histories.

Exactly preserves the current full-fold policy semantics. Never modifies the
runtime or consumes hidden fields. Object-identity guards reset for new copied,
truncated, or different-game histories; official history events are append-only.
This cache is a simulation speed tool, not a new inference belief claim.
"""
from uno_policy_v2 import symbol
from uno_route_policy import Policy as RoutePolicy,action_features

class PublicFold:
    def __init__(self):self.reset()

    def reset(self):
        self.lack_colour={};self.lack_symbol={};self.preference={};self.pending_draw={}
        self.active_colour=None;self.top_symbol=None;self.pending_penalty=0
        self.count=0;self.first=None;self.last=None

    def consume(self,events):
        lack_colour=self.lack_colour;lack_symbol=self.lack_symbol
        preference=self.preference;pending_draw=self.pending_draw
        active_colour=self.active_colour;top_symbol=self.top_symbol;pending_penalty=self.pending_penalty
        for event in events:
            kind, seat = (event.get('type'), event.get('seat'))
            if kind in ('game_start', 'initial_discard'):
                card = event.get('card', event.get('initial_discard', {})) or {}
                active_colour = card.get('color', active_colour)
                top_symbol = symbol(card)
            elif kind == 'initial_color_choice':
                active_colour = event['color']
                preference[seat] = active_colour
            elif kind in ('draw', 'penalty_draw'):
                count = event.get('drawn', 0) if kind == 'draw' else event.get('count', 0)
                for key in list(lack_colour):
                    if key[0] == seat:
                        lack_colour[key] *= 0.75 ** count
                for key in list(lack_symbol):
                    if key[0] == seat:
                        lack_symbol[key] *= 0.92 ** count
                if count:
                    preference.pop(seat, None)
                if kind == 'draw':
                    if active_colour:
                        lack_colour[seat, active_colour] = 0.75 ** count
                    if top_symbol:
                        lack_symbol[seat, top_symbol] = 0.92 ** count
                    pending_draw[seat] = (event.get('turn_number'), active_colour, top_symbol)
                else:
                    pending_penalty = 0
            elif kind in ('keep_auto', 'play'):
                card = event.get('card') or {}
                if kind == 'play':
                    colour, played_symbol = (card.get('color'), symbol(card))
                    if colour:
                        lack_colour[seat, colour] = 0.0
                    if played_symbol:
                        lack_symbol[seat, played_symbol] = 0.0
                drawn = pending_draw.get(seat)
                if drawn and drawn[0] == event.get('turn_number') and (kind == 'keep_auto' or event.get('from_draw')):
                    if drawn[1]:
                        lack_colour[seat, drawn[1]] = 1.0
                    if drawn[2]:
                        lack_symbol[seat, drawn[2]] = 1.0
                    pending_draw.pop(seat, None)
                if kind == 'play':
                    active_colour = event.get('active_color', card.get('color'))
                    top_symbol = symbol(card)
                    if card.get('kind') in ('wild', 'wild_draw_four'):
                        preference[seat] = active_colour
                    if 'pending_penalty' in event:
                        pending_penalty = event['pending_penalty']
        self.active_colour=active_colour;self.top_symbol=top_symbol;self.pending_penalty=pending_penalty

    def update(self,observation):
        history=observation.get('history',[])
        if self.count and (len(history)<self.count or not history or history[0] is not self.first or history[self.count-1] is not self.last):
            self.reset()
        self.consume(history[self.count:])
        self.count=len(history);self.first=history[0] if history else None;self.last=history[-1] if history else None
        return {'colours':dict(self.lack_colour),'symbols':dict(self.lack_symbol),'preference':dict(self.preference),'penalty':self.pending_penalty}


class CachedRoutePolicy(RoutePolicy):
    def __init__(self,weights,fold=None):
        super().__init__(weights);self.fold=fold if fold is not None else PublicFold()

    def select(self,observation):
        legal=observation['legal_actions']
        if len(legal)==1:return legal[0]['action_id']
        beliefs=self.fold.update(observation);cache={}
        return max(legal,key=lambda action:sum(w*x for w,x in zip(self.weights,action_features(observation,action,beliefs,cache))))['action_id']
