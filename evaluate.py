# evaluate.py
# Flies the same random scenarios in three modes and reports landing error:
#   1. no correction   2. crab correction   3. crab correction + GRU final approach
# Usage: python evaluate.py [flights_per_seed]   (default 50 → 100 flights, ~10 min)

import random
import statistics
import sys

import simulation as sim
import rl_model
from train import random_wind, random_positions

SEEDS = (1, 2)


def fly_once(start, target, corrections, use_ai):
    sim.CORRECTIONS  = corrections
    sim.USE_RL_MODEL = use_ai
    drone = sim.create_drone()
    sim.program_drone(drone, start, target)
    sim.compute_flight_plan(drone)
    sim.fly(drone)
    return drone


def main():
    per_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    rl_model.load_model("rl_weights.pt")

    errors = {"No correction": [], "Crab correction": [], "Crab correction + GRU final approach": []}
    distances = []

    for seed in SEEDS:
        random.seed(seed)
        for _ in range(per_seed):
            random_wind()
            start, target = random_positions()

            drone = fly_once(start, target, "OFF", False)
            errors["No correction"].append(sim.get_distance_to_target(drone))

            drone = fly_once(start, target, "ON", True)
            errors["Crab correction"].append(sim.get_distance_to_target(drone))
            sim.final_approach(drone)
            errors["Crab correction + GRU final approach"].append(sim.get_distance_to_target(drone))

            distances.append(sim.initial_distance_to_target(drone))
            print(f"flight {len(distances):>3}: " +
                  "  ".join(f"{e[-1]:6.1f}" for e in errors.values()), flush=True)

    n = len(distances)
    print(f"\n{n} flights, mean distance {statistics.mean(distances):.0f} m\n")
    print(f"{'mode':<38}{'mean':>8}{'median':>9}{'≤25 m':>9}")
    for mode, e in errors.items():
        within = sum(x <= 25 for x in e)
        print(f"{mode:<38}{statistics.mean(e):>7.0f}m{statistics.median(e):>8.0f}m{within:>5}/{n}")


if __name__ == "__main__":
    main()
