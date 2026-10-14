# UNO development candidate provenance

Team: TungMasterSoloClear (TECHTEAM11600). Source repository:
https://github.com/LeviathanAG/triad-uno. This is a development candidate;
the final capture commit and image digest have not been submitted.

The protocol, SDK starter, and local training simulator come from the organizer's
unoSDK at `0cda3acc1bb5c05fa1803a8fbd7761770dc6d5fe`. Their vendor attribution and
manifests remain intact. The action features, policy optimization, and inference
wrapper were developed with Codex during the event. No external pretrained model,
LLM runtime, opening dataset, or opponent's submission is incorporated.
Vendored protocol components are `src/uno_agent/protocol.py` and
`src/uno_agent/schemas/*.json`, with their exact origin recorded in
`src/uno_agent/VENDOR.json`. No organizer game logic is copied into the runtime.

`models/policy.json` contains 18 learned action-scoring weights. Training used
cross-entropy policy optimization with population 32 for 30 generations and 64
simulated matches per candidate. Policies receive only filtered observations.
Training, selection and final evaluation seeds are distinct; source seeds and
simulator state are not inputs to the deployed model.

The private Kaggle run is
https://www.kaggle.com/code/xdantalianx/triad-tactics-training (version 1).
Training dependencies include NumPy; an additional PyTorch/CUDA score-distillation
experiment ran on Kaggle T4. The distilled checkpoint was not selected for this
candidate. Deployment uses Python's standard library only, loads frozen weights
once, requires no network and does not import the local evaluator.

`models/evaluation.json` records the separate 512-game tests per opponent regime.
These results assess the tested baselines only; they do not establish event
winning probability or official admission.

Training source is under `training/`. Set `UNO_SDK` to this repository when
running `training/train_uno.py` with NumPy installed. GPU distillation additionally
requires PyTorch and `UNO_OUTPUT` pointing to the teacher checkpoint directory.
