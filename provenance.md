# UNO development candidate provenance

Team: TungMasterSoloClear (TECHTEAM11600). Private source repository:
https://github.com/LeviathanAG/triad-uno. This development candidate has not been
submitted to the event; final capture commit and image digest remain unset.

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

Private belief-rollout efficiency experiment (not deployed): audited snapshots
matched 7,808 filtered observations and 300 transitions. Incremental public
fold matched 18,892 folds and 4,691 route choices, including reset cases.
The fold cache assumes the pinned SDK appends events without changing old
events. In 232 fixed synthetic-world rollouts, identical returns took 1.85x
less time with both speed paths. Prepared sampling preserves the original
public-only samples and RNG state. No actual hidden cards or real game seed
is supplied to the probe. Official referee imports are development-only.

Fresh 112M development comparison (256 full games per regime) selected fast16
by league placement points with the incumbent eligible. League gain +0.015625
(95% paired seed-block bootstrap CI -0.003906 to +0.039063); starter +0.027344
(-0.007813 to +0.066406); random -0.019531 (-0.066406 to +0.019531). These
small, mixed results do not justify changing the retained runtime. The uniform
eight-style hypothetical rival mixture did not win selection. Approximate
sampled-model Gaussian override bounds do not guarantee real-game strength.
A fixed fast16 fresh 113M gate is running privately on free Kaggle CPU as
xdantalianx/triad-uno-belief-fast16 version 1, 512 four-seat blocks per unique
regime (league/starter/random). Maximum 16 trials, minimum 4, budget 0.15s,
all hypothetical rivals use retained route. Actual opponent identities are
unknown. Tonight's frozen test-round source 5474ba6 remains unchanged.
Restricted development-only Docker check (1 CPU, 1 GiB, read-only, network
none, pids 128): 16 games / 1,438 decisions; maximum 0.153045s decision and
26,402,816 bytes peak RSS. Zero hypothetical-world legality rejections.
The 115.1M resource fixture is not a policy strength selection family.
