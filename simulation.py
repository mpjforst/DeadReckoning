# simulation.py
import math
import random
from opensimplex import noise2 as pnoise2
# from drone_controller import get_corrections



# ─────────────────────────────────────────────
# CANVAS
# ─────────────────────────────────────────────

CANVAS_W = 1000
CANVAS_H = 1000

# ─────────────────────────────────────────────
# WIND PARAMS
# ─────────────────────────────────────────────

WIND_MEAN_DIRECTION      = 167
WIND_MEAN_SPEED          = 3
WIND_VARIATION           = 0.9
WIND_DIRECTION_VARIATION = 100
WIND_SCALE               = 0.005
WIND_OCTAVES             = 5

# ─────────────────────────────────────────────
# PROGRAM DRONE HERE
# ─────────────────────────────────────────────

STARTING_POSITION      = [768, 104]
TARGET_POSITION        = [105, 809]
DRONE_STARTING_BEARING = 270
CORRECTIONS             = "ON"          # TURN on / off LIVE CORRECTIONS to mitigate WIND OFFSET
USE_RL_MODEL            = True

# ─────────────────────────────────────────────
# DRONE SPECS
# ─────────────────────────────────────────────

DRONE_MAX_SPEED   = 10
WIND_SENSOR_NOISE = 0
COMPASS_NOISE     = 0
MAX_TURN_RATE     = 10.0   # degrees per second


# ─────────────────────────────────────────────
# SIMULATION PARAMS
# ─────────────────────────────────────────────

TIME_INCREMENT = 0.005




# ─────────────────────────────────────────────
# SIMULATION CORE
# ─────────────────────────────────────────────

def create_drone():
    drone = {}

    # PRE-PROGRAMMED (input before flight)
    drone["target_location"]           = []
    drone["starting_location"]         = []

    # ON-BOARD SENSORS (read during flight)
    drone["compass_reading"]           = 0.0    # MEASURED bearing — what drone actually reads
    drone["air_velocity"]              = 0.0    # airspeed sensor, relative to air not ground
    drone["apparent_wind_speed"]       = 0.0    # anemometer
    drone["apparent_wind_angle"]       = 0.0    # wind
    drone["elapsed_time"]              = 0.0    # clock

    # COMPUTED AND UPDATED ON-BOARD (derived from sensors)
    drone["flightplan"]                = {}
    drone["starting_bearing"]          = 0
    drone["initial_bearing_to_target"] = 0
    drone["remaining_flight_time"]     = 0.0
    drone["absolute_wind_x_at_drone_location"] = 0.0
    drone["absolute_wind_y_at_drone_location"] = 0.0
    drone["distance_covered_against_ground"] = 0
    drone["air_velocity_log"]       = []
    drone["apparent_wind_angle_log"] = []
    drone["apparent_wind_speed_log"] = []
    drone["bearingpath"]               = []

    # DISPLAY ONLY (ground truth, never accessed by controller)
    drone["bearing"]                   = 0.0    # TRUE bearing — set by physics, simulator only
    drone["current_location"]          = []
    drone["location_log"]              = []
    drone["rnn_hidden"]                = None

    return drone

def program_drone(drone, starting_loc, target_loc):
    drone["starting_location"] = list(starting_loc)
    drone["current_location"]  = list(starting_loc)
    drone["bearing"]           = DRONE_STARTING_BEARING
    drone["starting_bearing"]  = DRONE_STARTING_BEARING
    drone["target_location"]   = list(target_loc)

def compute_flight_plan(drone):
    distance_to_target    = initial_distance_to_target(drone)
    flight_time_to_target = distance_to_target / DRONE_MAX_SPEED
    drone_vector          = degrees_to_vector(drone["starting_bearing"])

    drone_x  = drone["starting_location"][0]
    drone_y  = drone["starting_location"][1]
    target_x = drone["target_location"][0]
    target_y = drone["target_location"][1]

    to_target_vector  = (target_x - drone_x, target_y - drone_y)
    degrees_to_target = get_degrees_between_two_vectors(drone_vector, to_target_vector)

    drone["flightplan"]["degrees_to_target"]     = degrees_to_target
    drone["flightplan"]["flight_time_to_target"] = flight_time_to_target
    drone["remaining_flight_time"] = flight_time_to_target

    return drone

def rotate_drone_towards_target(drone):
    new_bearing = (drone["bearing"] + drone["flightplan"]["degrees_to_target"]) % 360
    drone["bearing"] = new_bearing
    read_compass(drone)
    drone["initial_bearing_to_target"] = drone["compass_reading"]

def fly(drone):

    # TURN TO TARGET:
    rotate_drone_towards_target(drone)

    # START FLYING:
    drone["air_velocity"] = DRONE_MAX_SPEED

    while drone["remaining_flight_time"] > 0:
        # ── SIMULATOR: physics
        drift_in_local_wind(drone)
        propell(drone)

        # ── SIMULATOR: update sensors
        read_wind_sensor(drone)
        read_compass(drone)

        # ── DRONE: on-board computing
        if CORRECTIONS == "ON":
            correct_bearing(drone)
            correct_flight_time(drone)

        # ── LOGGING
        drone["apparent_wind_speed_log"].append(drone["apparent_wind_speed"])
        drone["remaining_flight_time"] -= TIME_INCREMENT
        drone["elapsed_time"]          += TIME_INCREMENT
        drone["location_log"].append(location(drone))
        drone["air_velocity_log"].append(drone["air_velocity"])
        drone["bearingpath"].append(drone["bearing"])
        drone["apparent_wind_angle_log"].append(drone["apparent_wind_angle"])

    return drone

# ─────────────────────────────────────────────
# PHYSICS
# ─────────────────────────────────────────────

def drift_in_local_wind(drone):
    dx, dy = ground_wind_at(drone["current_location"][0], drone["current_location"][1])
    drone["current_location"][0] += dx * TIME_INCREMENT
    drone["current_location"][1] += dy * TIME_INCREMENT

def propell(drone):
    distance = drone["air_velocity"] * TIME_INCREMENT
    dx, dy   = vector_from_length_and_degrees(distance, drone["bearing"])
    drone["current_location"][0] += dx
    drone["current_location"][1] += dy
    return drone

def ground_wind_at(x, y):
    def octave_noise(x, y, octaves):
        val     = 0.0
        amp     = 1.0
        freq    = 1.0
        max_val = 0.0
        for _ in range(octaves):
            val     += pnoise2(x * freq, y * freq) * amp
            max_val += amp
            amp     *= 0.5
            freq    *= 2.0
        return val / max_val

    speed_noise     = octave_noise(x * WIND_SCALE,         y * WIND_SCALE,         WIND_OCTAVES)
    direction_noise = octave_noise(x * WIND_SCALE + 100.0, y * WIND_SCALE + 100.0, WIND_OCTAVES)

    speed     = max(0.0, WIND_MEAN_SPEED + speed_noise * WIND_MEAN_SPEED * WIND_VARIATION)
    direction = WIND_MEAN_DIRECTION + direction_noise * WIND_DIRECTION_VARIATION

    return vector_from_length_and_degrees(speed, direction)

# ─────────────────────────────────────────────
# CORRECTIONS
# ─────────────────────────────────────────────

def correct_bearing(drone):
    target_bearing  = drone["initial_bearing_to_target"]
    airspeed        = drone["air_velocity"]

    # ── apparent wind vector from vane (drone-observable only)
    apparent_speed  = drone["apparent_wind_speed"]
    apparent_angle  = drone["apparent_wind_angle"]

    apparent_wx = -math.cos(math.radians(apparent_angle)) * apparent_speed
    apparent_wy = -math.sin(math.radians(apparent_angle)) * apparent_speed

    # ── recover true wind = apparent wind + drone velocity
    vx, vy  = vector_from_length_and_degrees(airspeed, drone["compass_reading"])
    true_wx = apparent_wx + vx
    true_wy = apparent_wy + vy

    drone["absolute_wind_x_at_drone_location"] = true_wx
    drone["absolute_wind_y_at_drone_location"] = true_wy


    # ── wind triangle: solve crab angle
    track_rad = math.radians(target_bearing)
    tx =  math.cos(track_rad)
    ty =  math.sin(track_rad)

    crosswind  = true_wx * (-ty) + true_wy * tx
    ratio      = max(-1.0, min(1.0, crosswind / airspeed))
    crab_angle = math.degrees(math.asin(ratio))

    desired_heading = (target_bearing - crab_angle) % 360     # BEARING THE DRONE NEEDS TO GO TO HIT TARGET BASED ON CURRENT WIND CONDITIONS (NOT BEARING TO TARGET) 

    delta = (desired_heading - drone["compass_reading"]) % 360
    if delta > 180:
        delta -= 360

    max_delta = MAX_TURN_RATE * TIME_INCREMENT 
    delta     = max(-max_delta, min(max_delta, delta))

    drone["bearing"]      = (drone["compass_reading"] + delta) % 360
    drone["air_velocity"] = DRONE_MAX_SPEED

def correct_flight_time(drone):
    # in the simple case: drone flies into headwind.
    # effective ground speed = airspeed - headwind.
    # each timestep covers less distance than expected, so add back the deficit.

    airspeed = drone["air_velocity"]

    wx = drone["absolute_wind_x_at_drone_location"]
    wy = drone["absolute_wind_y_at_drone_location"]

    track_rad = math.radians(drone["initial_bearing_to_target"])
    tx, ty    = math.cos(track_rad), math.sin(track_rad)

    headwind = -(wx * tx + wy * ty)   # positive = opposing motion

    effective_ground_speed = airspeed - headwind

    if effective_ground_speed > 0:
        deficit = (airspeed - effective_ground_speed) / airspeed
        drone["remaining_flight_time"] += deficit * TIME_INCREMENT

def final_approach(drone):
    if not USE_RL_MODEL:
        return

    from rl_model import predict
    delta_bearing, extra_time = predict(drone)

    if extra_time <= 0:
        return

    # set new heading and update reference bearing for corrections
    drone["bearing"]                   = (drone["compass_reading"] + delta_bearing) % 360
    drone["remaining_flight_time"]     = extra_time
    drone["initial_bearing_to_target"] = drone["bearing"]

    while drone["remaining_flight_time"] > 0:
        drift_in_local_wind(drone)
        propell(drone)
        read_wind_sensor(drone)
        read_compass(drone)

        # if CORRECTIONS == "ON":
        #     correct_bearing(drone)
        #     correct_flight_time(drone)

        drone["remaining_flight_time"] -= TIME_INCREMENT
        drone["elapsed_time"]          += TIME_INCREMENT
        drone["location_log"].append(location(drone))
        drone["air_velocity_log"].append(drone["air_velocity"])
        drone["bearingpath"].append(drone["bearing"])
        drone["apparent_wind_angle_log"].append(drone["apparent_wind_angle"])
        drone["apparent_wind_speed_log"].append(drone["apparent_wind_speed"])

# ─────────────────────────────────────────────
# ON-BOARD DRONE SENSORS
# ─────────────────────────────────────────────

def onboard_wind_sensor(drone):
    # SIMULATOR ONLY — simulates what the physical sensor would measure
    # drone uses this as input

    # GET GROUND WIND:

    (x, y) = location(drone)
    wx, wy = ground_wind_at(x, y)
    vx, vy = vector_from_length_and_degrees(drone["air_velocity"], drone["bearing"])           

    # COMPUTE APPARENT WIND: 

    apparent_wx = wx - vx
    apparent_wy = wy - vy
    apparent_speed = math.hypot(apparent_wx, apparent_wy)

    if apparent_speed > 0.001:
        apparent_angle = math.degrees(math.atan2(-apparent_wy, -apparent_wx)) % 360
    else:
        apparent_angle = drone["apparent_wind_angle"]  # hold last reading


    if WIND_SENSOR_NOISE > 0:
        apparent_angle += random.gauss(0, WIND_SENSOR_NOISE)
        apparent_speed += random.gauss(0, WIND_SENSOR_NOISE)

    return apparent_speed, apparent_angle

def read_wind_sensor(drone):
    speed, angle = onboard_wind_sensor(drone)

    drone["apparent_wind_speed"]       = speed    
    drone["apparent_wind_angle"]       = angle    

def onboard_compass(drone):
    if COMPASS_NOISE > 0:
        drone["compass_reading"] = (drone["bearing"] + random.gauss(0, COMPASS_NOISE)) % 360
    else: 
        drone["compass_reading"] = drone["bearing"]

def read_compass(drone):
    onboard_compass(drone)
    return drone["compass_reading"]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def vector_from_length_and_degrees(length, degrees):
    r = math.radians(degrees)
    return round(math.cos(r) * length, 10), round(math.sin(r) * length, 10)

def get_degrees_between_two_vectors(v1, v2):
    angle1 = math.atan2(v1[1], v1[0])
    angle2 = math.atan2(v2[1], v2[0])
    return math.degrees(angle2 - angle1) % 360

def degrees_to_vector(degrees):
    r = math.radians(degrees)
    return (round(math.cos(r), 10), round(math.sin(r), 10))

def initial_distance_to_target(drone):
    tx, ty = drone["target_location"]
    sx, sy = drone["starting_location"]
    return math.sqrt((tx - sx)**2 + (ty - sy)**2)

def get_distance_to_target(drone):
    tx, ty = drone["target_location"]
    dx, dy = drone["current_location"]
    return math.sqrt((tx - dx)**2 + (ty - dy)**2)

def location(drone):
    x = drone["current_location"][0]
    y = drone["current_location"][1]
    return (x,y)

# ─────────────────────────────────────────────
# MAIN (single run, no display)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if USE_RL_MODEL:
        from rl_model import load_model
        load_model("rl_weights.pt")
    drone = create_drone()
    program_drone(drone, STARTING_POSITION, TARGET_POSITION)
    compute_flight_plan(drone)
    fly(drone)

    error_before = get_distance_to_target(drone)
    final_approach(drone)
    error_after = get_distance_to_target(drone)

    print(f"Landing location  : ({drone['current_location'][0]:.1f}, {drone['current_location'][1]:.1f})")
    print(f"Error before final approach: {error_before:.1f} m")
    print(f"Error after  final approach: {error_after:.1f} m")
    print(f"Improvement                : {error_before - error_after:.1f} m")

    from display import animate_flightpath
    animate_flightpath(drone)