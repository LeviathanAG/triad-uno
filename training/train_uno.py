"""Cross-entropy policy optimization on real SDK games, followed by held-out evaluation.

Training uses filtered observations. Simulation seeds are local, and are never
passed to the learned policy. The exported policy has no training dependencies.
"""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
SDK = Path(os.environ.get("UNO_SDK", ROOT.parent / "unoSDK"))
sys.path.insert(0, str(SDK / "local_evaluator" / "src"))
from triad_uno.baselines import HeuristicAgent, RandomAgent
from triad_uno.engine import GameConfig
from triad_uno.simulator import run_game
from uno_policy import INITIAL_WEIGHTS, FEATURE_NAMES, Policy


def evaluate_job(job):
    weights, seeds, regime = job
    points, wins = [], 0
    for index, seed in enumerate(seeds):
        seat = index % 4
        opponents = []
        for other in range(4):
            if other == seat:
                opponents.append(Policy(weights))
            elif regime == "heuristic" or (regime == "mixed" and (seed + other) % 3 == 0):
                opponents.append(HeuristicAgent(seed * 10 + other))
            elif regime == "random" or (regime == "mixed" and (seed + other) % 3 == 1):
                opponents.append(RandomAgent(seed * 10 + other))
            else:
                opponents.append(Policy(INITIAL_WEIGHTS))
        final, _, _ = run_game(seed, opponents, GameConfig(max_turns=1000))
        outcome = final.outcome
        if seat in outcome.placements:
            placement = outcome.placements.index(seat)
            points.append((5., 3., 2., 1.)[placement])
            wins += int(placement == 0)
        else:
            remaining = (5., 3., 2., 1.)[len(outcome.placements):]
            points.append(sum(remaining) / max(1, len(outcome.unfinished)))
    return {"points": float(np.mean(points)), "win_rate": wins / len(seeds), "games": len(seeds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--eval-games", type=int, default=256)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    if min(args.iterations, args.population, args.games, args.eval_games, args.workers) < 1:
        parser.error("all counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    mean = np.asarray(INITIAL_WEIGHTS, dtype=float)
    sigma = np.asarray([2.0] * len(mean))
    started = time.time()
    history = []
    best = mean.copy()
    # Validation seeds are distinct from every generation's training seeds.
    validation_seeds = list(range(2_000_000, 2_000_000 + args.eval_games))
    selected_validation = -float("inf")
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for generation in range(args.iterations):
            population = rng.normal(mean, sigma, (args.population, len(mean)))
            population[0] = mean
            if args.population > 1:
                population[1] = best
            seeds = list(range(10_000 + generation * args.games, 10_000 + (generation + 1) * args.games))
            scores = list(pool.map(evaluate_job, [(w.tolist(), seeds, "mixed") for w in population]))
            ranked = sorted(range(len(scores)), key=lambda i: (scores[i]["points"], scores[i]["win_rate"]), reverse=True)
            elite = population[ranked[:max(2, args.population // 4)]]
            mean = .3 * mean + .7 * elite.mean(axis=0)
            sigma = np.maximum(.15, .3 * sigma + .7 * elite.std(axis=0))
            candidate = population[ranked[0]]
            validation = list(pool.map(evaluate_job, [(candidate.tolist(), validation_seeds, "mixed")]))[0]
            if validation["points"] > selected_validation:
                best = candidate.copy()
                selected_validation = validation["points"]
            entry = {"generation": generation + 1, "training": scores[ranked[0]], "validation": validation,
                     "elapsed_seconds": round(time.time() - started, 1)}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            metadata = {"method": "cross-entropy policy optimization", "seed": args.seed,
                        "generations": generation + 1, "population": args.population,
                        "games_per_candidate": args.games, "validation": selected_validation,
                        "sdk_commit": "0cda3acc1bb5c05fa1803a8fbd7761770dc6d5fe",
                        "feature_count": len(FEATURE_NAMES)}
            Policy(best).save(args.output / "uno-policy.json", metadata)
            (args.output / "training-progress.json").write_text(json.dumps(history, indent=2) + "\n")
        # Final test seeds were not used for selection. Run three opponent regimes.
        final_seeds = list(range(3_000_000, 3_000_000 + args.eval_games))
        jobs = [(w, final_seeds, regime) for regime in ("heuristic", "random", "mixed")
                for w in (best.tolist(), INITIAL_WEIGHTS)]
        results = list(pool.map(evaluate_job, jobs))
        report = {regime: {"learned": results[2*i], "initial": results[2*i+1]}
                  for i, regime in enumerate(("heuristic", "random", "mixed"))}
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"held_out_evaluation": report}), flush=True)


if __name__ == "__main__":
    main()
