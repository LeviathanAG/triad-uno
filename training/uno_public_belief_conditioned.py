"""Isolated exact one-seat-conditioned rejection under the uniform prior.

Uniformly choose an ordered hand from the anchor's allowed physical cards,
then uniformly permute every remaining physical card. This is the original
uniform proposal conditioned on the anchor's exclusion event. Rejecting other
seats therefore yields the same joint hard-constraint conditional law. The
anchor is chosen deterministically from the public observation alone.

This improves proposal acceptance, not the uniform prior's behavioural accuracy.
No real hidden state or private game seed is accepted. Current live jobs retain
the original sampler. Multi-seat sequential conditioning without weights would
generally bias the posterior and is deliberately not used here.
"""
import math
from uno_public_belief import PreparedSampler, compatible


class PreparedConditionedSampler(PreparedSampler):
    def __init__(self,observation):
        super().__init__(observation)
        self._prepare_anchor()

    def _prepare_anchor(self):
        n=len(self.cards);choices=[]
        for seat,count in enumerate(self.sizes):
            if seat==self.seat or count==0:continue
            allowed=[i for i,card in enumerate(self.cards)
                     if compatible(card,self.constraints[seat])]
            if len(allowed)<count:
                choices.append((float('-inf'),seat,allowed));continue
            # log(C(allowed,count)/C(total,count)); physical copies are distinct.
            log_probability=sum(math.log((len(allowed)-i)/(n-i)) for i in range(count))
            choices.append((log_probability,seat,allowed))
        self.anchor=None;self.allowed=[]
        if choices:
            log_probability,self.anchor,self.allowed=min(choices,key=lambda x:(x[0],x[1]))
            if log_probability==0:self.anchor=None

    def sample(self,rng,max_attempts=32):
        if self.anchor is None:return super().sample(rng,max_attempts)
        count=self.sizes[self.anchor]
        if len(self.allowed)<count:return None
        for _ in range(max_attempts):
            selected=rng.sample(self.allowed,count);used=set(selected)
            remaining=[card for i,card in enumerate(self.cards) if i not in used]
            rng.shuffle(remaining)
            hands={self.anchor:[self.cards[i] for i in selected]};offset=0;ok=True
            for other,size in enumerate(self.sizes):
                if other in (self.seat,self.anchor):continue
                hand=remaining[offset:offset+size];offset+=size
                if not all(compatible(card,self.constraints[other]) for card in hand):
                    ok=False;break
                hands[other]=hand
            if ok:return hands,remaining[offset:]
        return None
