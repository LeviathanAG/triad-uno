"""Filtered public-history evidence; no hidden simulator inputs."""

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
