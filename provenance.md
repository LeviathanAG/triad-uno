# UNO test-round candidate provenance

Team: TungMasterSoloClear (TECHTEAM11600). Private source repository:
https://github.com/LeviathanAG/triad-uno. The October 14 test-round source is prepared for the authorized 20:30 IST push.
The retained runtime/model source is verified at
`715e74042973c99838b537e7fac027fed63ba64a`; the complete prepared source commit
and any published image digest are recorded in the publication receipt. The
registry/upload destination is pending organizer confirmation; no image URI is
fabricated in the manifest.

The organizer UNO simulator used for training is the exact snapshot
`0cda3acc1bb5c05fa1803a8fbd7761770dc6d5fe`. The later SDK build/checker update
`a1e9592407c816edc51fce024d7344516c8d3074` is merged and does not change game
logic. Organizer protocol/vendor files retain their attribution and manifests.
No local evaluator or copied organizer engine is imported by deployed inference.

The policy is an original non-language linear action scorer using Python's
standard library. It has 24 weights: 18 frozen weights from the previous learned
policy, plus six newly trained residual weights. Own-hand sequence features use
an original deterministic search capped at 128 visited states and depth eight.
The search imagines consecutive own plays with no intervening opponent changes;
its result is a hand-structure feature, not a guaranteed future sequence.
Other features use public history to infer colour/symbol absences and preferences.
Inference reads only the filtered observation and recomputes history evidence on
every request. It consumes no hidden hands, simulator state, seed, or game ID.

The initial 18-weight model was trained by population cross-entropy optimization
in https://www.kaggle.com/code/xdantalianx/triad-tactics-training (version 1).
It is retained as `models/incumbent-v1.json`, with its original evaluation report.
The additional six weights were trained in the private CPU notebook
https://www.kaggle.com/code/xdantalianx/triad-uno-route-residual (version 1):
24 generations, population 24, 24 four-seat seed blocks per candidate. Finalists
were compared once on a fixed 256-block selection set. The first final test used
512 fresh blocks per opponent regime. Training source, selection and progress
are under `training/`; NumPy is a training dependency, never a runtime dependency.
Set `UNO_SDK` to this repository and pass `models/incumbent-v1.json` to reproduce
residual training. No external pretrained model or participant agent is used.

`models/evaluation.json` reports 2,048 games per regime. The varied league gain
was +0.07275 points per game (paired 95% interval +0.02588 to +0.11914).
`models/independent-evaluation.json` reports another 4,096 games per regime from
fresh seed 26,000,000 with the chosen checkpoint fixed. That replication gave a
smaller, uncertain league gain +0.00977 (interval -0.02319 to +0.04224), a
starter/heuristic gain +0.04663 (interval +0.01563 to +0.07861), and a random-rival
gain +0.03223 (interval -0.00245 to +0.06812). All tested point means improved;
these local results establish neither event winning probability nor universal
superiority. League first-place rate was slightly lower on the replication.

Validation: 300 exhaustive small-hand checks against authoritative SDK legality;
2,423 zero-residual compatibility/legality decisions; packaged inference matches
the training policy across 1,649 decisions in 16 complete games. SDK tests,
protocol smoke checks, three local restart games, and three network-disabled,
read-only Docker restart games pass with zero agent faults or fallbacks.
The CPU image continues to use the ordinary Dockerfile and claims no shared
PyTorch/CUDA size exemption. Later private neural experiments remain separate
until they demonstrate a gain over this retained checkpoint.

The test-round manifest identifies team TungMasterSoloClear / TECHTEAM11600 and
declares the actual loaded `models/policy.json` asset (SHA-256
`d458ef2894c0ce51d62ed30a4ad1b685c85db2c1a97417a8a93efab16d5abe14`).
The CPU parent restores the official literal `FROM python:3.11-slim` declaration
for organizer compatibility. The exact tested index and Linux/amd64 platform
digests are recorded in the admission verification receipt; the tag is mutable.
Current admission preparation repeats the eight protocol checks, the SDK suite
(27 passed, six optional checks skipped), and three local plus three restricted
Docker restart games, with zero unintended faults or fallbacks. The test image
uses UID10000:10000, no network, read-only root, one CPU, one GiB RAM, and bounded
scratch. Required SDK resource labels remain unchanged. Later joint24 and
mirrored searches failed to establish an upgrade; the shipped route policy is
unchanged from the verified source above.

A bounded original certificate handles exactly two unfinished seats. If a currently legal Skip begins an own-hand sequence of only Skips followed by at most one matching final card, no opponent can act before that hand empties. The agent may choose this guaranteed finish over the scalar score only with enough placement turns before the public configured limit. It stores validated initialize.config.max_turns in its permitted JSON checkpoint and subtracts current turns_completed; absent/invalid configuration declines to the existing scorer. References are recomputed from the current observation; no hidden opponent state, shuffle seed, opponent model, or search library is used. This can improve third-versus-fourth placement only and does not repair the first-place-rate gap. The trained24-weight asset and required teamName remain unchanged.
