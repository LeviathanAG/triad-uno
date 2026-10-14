"""Train a compact GPU policy from the optimized teacher on real simulated observations."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ["UNO_SDK"]) / "local_evaluator" / "src"))
from triad_uno.simulator import run_game
from triad_uno.engine import GameConfig
from uno_policy import Policy, action_features, FEATURE_NAMES

OUTPUT = Path(os.environ.get("UNO_OUTPUT", "/kaggle/working/output"))
teacher_document = json.loads((OUTPUT / "uno-policy.json").read_text())
teacher = Policy(teacher_document["weights"])
samples = []


class Collector:
    def select(self, observation):
        if len(observation["legal_actions"]) > 1:
            samples.extend(action_features(observation, action) for action in observation["legal_actions"])
        return teacher.select(observation)


for seed in range(4_000_000, 4_000_512):
    run_game(seed, [Collector() for _ in range(4)], GameConfig(max_turns=1000))
x = np.asarray(samples, dtype=np.float32)
y = x @ np.asarray(teacher.weights, dtype=np.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(20261014)
torch.set_num_threads(2)
features = torch.tensor(x, device=device)
targets = torch.tensor(y, device=device).unsqueeze(1)
order = torch.randperm(len(features), device=device)
split = int(.9 * len(order))
train, validation = order[:split], order[split:]
model = torch.nn.Linear(len(FEATURE_NAMES), 1, bias=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=.03)
for step in range(2000):
    indices = train[torch.randint(len(train), (min(8192, len(train)),), device=device)]
    loss = torch.nn.functional.mse_loss(model(features[indices]), targets[indices])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(json.dumps({"distillation_step": step, "mse": float(loss.detach()), "device": str(device)}), flush=True)
with torch.no_grad():
    validation_rmse = float(torch.mean((model(features[validation]) - targets[validation]) ** 2).sqrt())
weights = model.weight.detach().cpu().flatten().tolist()
metadata = {**teacher_document["training"], "distillation": {"method": "supervised score distillation",
    "samples": len(features), "steps": 2000, "device": str(device), "validation_rmse": validation_rmse,
    "torch_version": torch.__version__, "training_data": "filtered observations from 512 real SDK self-play games"}}
Policy(weights).save(OUTPUT / "uno-policy-distilled.json", metadata)
print(json.dumps(metadata["distillation"]), flush=True)
