"""Isolated exact-speed experiment for the bounded own-hand route feature.

Traversal, memo keys, depth and visit budgets preserve the existing feature.
This module is not wired into runtime or the running fixed belief gate.
"""
from collections import Counter
from functools import lru_cache


def route_length(hand,colour,top,max_states=128,max_depth=8):
    cards=tuple(sorted(Counter((c.get('color') or '',c['kind'],
               c.get('value') if c.get('value') is not None else -1)
               for c in hand).items()))
    types=tuple(c for c,_ in cards);initial=tuple(n for _,n in cards)
    colours=sorted({c for c,_,_ in types if c})
    colour_indices={c:tuple(i for i,t in enumerate(types) if t[0]==c)
                    for c in colours}
    wilds=tuple(kind in ('wild','wild_draw_four') for _,kind,_ in types)
    visits=0

    @lru_cache(maxsize=256)
    def walk(counts,active,last_kind,last_value,depth):
        nonlocal visits
        if not depth or visits>=max_states:return 0
        visits+=1;best=0;goal=min(depth,sum(counts))
        has_active=any(counts[i] for i in colour_indices.get(active,()))
        for i,((c,kind,value),n) in enumerate(zip(types,counts)):
            if not n:continue
            legal=(kind=='wild' or (kind=='wild_draw_four' and not has_active)
                   or c==active or (kind=='number' and last_kind=='number' and value==last_value)
                   or kind in ('skip','reverse','draw_two') and kind==last_kind)
            if not legal:continue
            reduced=counts[:i]+(n-1,)+counts[i+1:]
            options=([other for other in colours
                      if any(reduced[j] for j in colour_indices[other])]
                     if wilds[i] else [c])
            if not options:options=[active]
            for chosen in options:
                length=1+walk(reduced,chosen,kind,value,depth-1)
                if length>best:best=length
                if best>=goal:return best
        return best
    return walk(initial,colour or '',top.get('kind'),top.get('value',-1),min(max_depth,len(hand)))
