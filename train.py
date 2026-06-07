# train_rl.py

import torch
import torch.nn as nn
import random
import math

from simulation import (
    create_drone, program_drone, compute_flight_plan,
    fly, get_distance_to_target, location,
    drift_in_local_wind, propell, read_wind_sensor, read_compass,
    CANVAS_W, CANVAS_H, TIME_INCREMENT, DRONE_MAX_SPEED
)
import simulation
import rl_model
from rl_model import FinalApproachRNN, build_input_tensor, save_model

# ─────────────────────────────────────────────
# TRAINING PARAMS
# ─────────────────────────────────────────────

NUM_EPISODES  = 2000
LEARNING_RATE = 1e-3
PRINT_EVERY   = 10
SAVE_PATH     = "rl_weights.pt"

# ─────────────────────────────────────────────
# RANDOM EPISODE SETUP
# ─────────────────────────────────────────────

def random_wind():
    simulation.WIND_MEAN_DIRECTION      = random.uniform(0, 360)
    simulation.WIND_MEAN_SPEED          = random.uniform(1, 6)
    simulation.WIND_VARIATION           = random.uniform(0.1, 0.6)
    simulation.WIND_DIRECTION_VARIATION = random.uniform(10, 60)
    simulation.WIND_SCALE               = random.uniform(0.001, 0.008)

def random_positions():
    margin = 100
    start  = [random.uniform(margin, CANVAS_W - margin),
              random.uniform(margin, CANVAS_H - margin)]
    target = [random.uniform(margin, CANVAS_W - margin),
              random.uniform(margin, CANVAS_H - margin)]
    return start, target


# ─────────────────────────────────────────────
# COMPUTE OPTIMAL CORRECTION (ground truth cheat — training only)
# ─────────────────────────────────────────────

def compute_optimal_correction(drone):
    """
    After fly() ends, peek at ground truth position and target.
    Compute exact bearing and extra time needed to reach target.
    This is the supervised label — drone never has access to this during inference.
    """
    cx, cy = drone["current_location"]
    tx, ty = drone["target_location"]

    dx = tx - cx
    dy = ty - cy
    distance = math.hypot(dx, dy)

    if distance < 0.1:
        return 0.0, 0.0

    # bearing from current location to target
    optimal_bearing = math.degrees(math.atan2(dy, dx)) % 360

    # delta from current compass reading
    delta_bearing = (optimal_bearing - drone["compass_reading"]) % 360
    if delta_bearing > 180:
        delta_bearing -= 360

    # time needed to cover remaining distance at max speed
    extra_time = distance / DRONE_MAX_SPEED

    return delta_bearing, extra_time


# ─────────────────────────────────────────────
# EPISODE
# ─────────────────────────────────────────────

def run_episode(model, criterion, optimizer):
    random_wind()
    start, target = random_positions()

    drone = create_drone()
    program_drone(drone, start, target)
    compute_flight_plan(drone)
    simulation.USE_RL_MODEL = False
    fly(drone)

    error_before = get_distance_to_target(drone)

    # ── ground truth label (training only cheat)
    delta_bearing_opt, extra_time_opt = compute_optimal_correction(drone)
    label = torch.tensor(
        [delta_bearing_opt / 180.0,   # normalise to -1..1
         extra_time_opt    / 100.0],  # normalise by rough max
        dtype=torch.float32
    ).unsqueeze(0)  # (1, 2)

    # ── model prediction from sensor history
    x = build_input_tensor(drone)
    prediction = model(x)   # (1, 2)

    # ── supervised loss
    optimizer.zero_grad()
    loss = criterion(prediction, label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # ── evaluate: apply predicted correction and fly
    with torch.no_grad():
        pred_delta_bearing = prediction[0, 0].item() * 180.0
        pred_extra_time    = max(0.0, prediction[0, 1].item() * 100.0)

    drone_eval = create_drone()
    program_drone(drone_eval, start, target)
    compute_flight_plan(drone_eval)
    simulation.USE_RL_MODEL = False
    fly(drone_eval)

    drone_eval["bearing"]               = (drone_eval["compass_reading"] + pred_delta_bearing) % 360
    drone_eval["remaining_flight_time"] = pred_extra_time

    while drone_eval["remaining_flight_time"] > 0:
        drift_in_local_wind(drone_eval)
        propell(drone_eval)
        read_wind_sensor(drone_eval)
        read_compass(drone_eval)
        drone_eval["remaining_flight_time"] -= TIME_INCREMENT
        drone_eval["elapsed_time"]          += TIME_INCREMENT

    error_after = get_distance_to_target(drone_eval)

    return loss.item(), error_before, error_after


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train():
    model     = FinalApproachRNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    rl_model._model = model

    print(f"training for {NUM_EPISODES} episodes...")
    print(f"{'episode':>10}  {'loss':>10}  {'before':>10}  {'after':>10}  {'improvement':>12}")
    print("-" * 60)

    recent_loss   = []
    recent_before = []
    recent_after  = []

    for episode in range(1, NUM_EPISODES + 1):
        model.train()
        loss, error_before, error_after = run_episode(model, criterion, optimizer)

        recent_loss.append(loss)
        recent_before.append(error_before)
        recent_after.append(error_after)
        if len(recent_loss) > 100:
            recent_loss.pop(0)
            recent_before.pop(0)
            recent_after.pop(0)

        if episode % PRINT_EVERY == 0:
            avg_loss   = sum(recent_loss)   / len(recent_loss)
            avg_before = sum(recent_before) / len(recent_before)
            avg_after  = sum(recent_after)  / len(recent_after)
            print(f"{episode:>10}  {avg_loss:>10.4f}  {avg_before:>10.1f}  "
                  f"{avg_after:>10.1f}  {avg_before - avg_after:>12.1f}")

    print(f"\n── final results ────────────────────────")
    print(f"avg error before : {sum(recent_before) / len(recent_before):.1f} m")
    print(f"avg error after  : {sum(recent_after)  / len(recent_after):.1f} m")
    print(f"improvement      : {(sum(recent_before) - sum(recent_after)) / len(recent_before):.1f} m")

    answer = input("save model? y/n: ")
    if answer.lower() == "y":
        save_model(SAVE_PATH)


if __name__ == "__main__":
    train()