"""Small non-language action scorer with a learned residual over the incumbent."""
import json
import math
from pathlib import Path
from uno_policy_v2 import FEATURE_NAMES, action_features, public_beliefs


class Policy:
    def __init__(self, model):
        self.feature_names = FEATURE_NAMES
        self.action_features = action_features
        if model.get('format') == 'triad-uno-residual-mlp-v3':
            from uno_policy_features_v4 import FEATURE_NAMES as rich_names, action_features as rich_features
            self.feature_names = rich_names
            self.action_features = rich_features
        elif model.get('format') == 'triad-uno-residual-mlp-v2':
            from uno_policy_features_v3 import FEATURE_NAMES as rich_names, action_features as rich_features
            self.feature_names = rich_names
            self.action_features = rich_features
        elif model.get('format') != 'triad-uno-residual-mlp-v1':
            raise ValueError('incompatible model format')
        if model.get('features') != self.feature_names:
            raise ValueError('incompatible model format/features')
        self.model = model
        self.base = model['base_weights']
        self.w1, self.b1 = model['hidden_weight'], model['hidden_bias']
        self.w2, self.b2 = model['output_weight'], model['output_bias']
        width = len(self.b1)
        if len(self.base) != len(self.feature_names) or len(self.w1) != width or len(self.w2) != width or not 1 <= width <= 128:
            raise ValueError('invalid model dimensions')
        if any(len(row) != len(self.feature_names) for row in self.w1):
            raise ValueError('invalid hidden dimensions')
        values = self.base + self.b1 + self.w2 + [self.b2] + [x for row in self.w1 for x in row]
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError('nonfinite model parameter')

    def score_features(self, vector):
        hidden = [max(0., sum(w*x for w,x in zip(row,vector)) + bias) for row,bias in zip(self.w1,self.b1)]
        return sum(w*x for w,x in zip(self.base,vector)) + sum(w*x for w,x in zip(self.w2,hidden)) + self.b2

    def select(self, observation):
        legal = observation['legal_actions']
        if len(legal) == 1:
            return legal[0]['action_id']
        beliefs = public_beliefs(observation)
        return max(legal,key=lambda action:self.score_features(self.action_features(observation,action,beliefs)))['action_id']

    def save(self, path):
        Path(path).write_text(json.dumps(self.model,indent=2)+'\n')


class FastPolicy(Policy):
    """NumPy batch scoring for training evaluation only; never shipped."""
    def __init__(self, model):
        super().__init__(model)
        import numpy as np
        self.np = np
        self.fast_base = np.asarray(self.base, dtype=float)
        self.fast_w1 = np.asarray(self.w1, dtype=float)
        self.fast_b1 = np.asarray(self.b1, dtype=float)
        self.fast_w2 = np.asarray(self.w2, dtype=float)

    def select(self, observation):
        legal = observation['legal_actions']
        if len(legal) == 1:
            return legal[0]['action_id']
        beliefs = public_beliefs(observation)
        x = self.np.asarray([self.action_features(observation,a,beliefs) for a in legal], dtype=float)
        hidden = self.np.maximum(0., x @ self.fast_w1.T + self.fast_b1)
        scores = x @ self.fast_base + hidden @ self.fast_w2 + self.b2
        return legal[int(self.np.argmax(scores))]['action_id']
