# display.py

import tkinter as tk
import math

from simulation import (
    CANVAS_W, CANVAS_H,
    WIND_MEAN_SPEED,
    ground_wind_at, get_distance_to_target,
    create_drone, program_drone, compute_flight_plan, fly, final_approach, initial_distance_to_target,
    DRONE_STARTING_BEARING
)
import simulation

# ─────────────────────────────────────────────
# DISPLAY PARAMS
# ─────────────────────────────────────────────

ARROW_GRID_SPACING = 50
ARROW_MAX_LENGTH   = 50
BG_COLOR           = "#f5f4ef"
DRONE_COLOR        = "#378add"
FLAG_COLOR         = "#d85a30"
TRAIL_COLOR        = "#85b7eb"
WIND_COLOR         = "#3a5a8a"
START_COLOR        = "#3a8a3a"
TARGET_COLOR       = "#d85a30"
FLAG_LENGTH        = 30


# ─────────────────────────────────────────────
# WIND FIELD
# ─────────────────────────────────────────────

def draw_wind_field(canvas):
    sample_pts = []
    for gx in range(0, CANVAS_W, ARROW_GRID_SPACING):
        for gy in range(0, CANVAS_H, ARROW_GRID_SPACING):
            wx, wy = ground_wind_at(gx, gy)
            spd    = math.hypot(wx, wy)
            sample_pts.append((gx, gy, wx, wy, spd))

    scale = ARROW_MAX_LENGTH / (WIND_MEAN_SPEED * 2)

    for gx, gy, wx, wy, spd in sample_pts:
        if spd < 0.01:
            continue
        ex = gx + wx * scale
        ey = gy + wy * scale
        canvas.create_line(gx, gy, ex, ey,
                           fill=WIND_COLOR, width=1,
                           arrow=tk.LAST, arrowshape=(4, 5, 2),
                           tags="wind_field")


# ─────────────────────────────────────────────
# MARKERS
# ─────────────────────────────────────────────

def draw_markers(canvas, drone):
    tx, ty = drone["target_location"]
    r = 9
    canvas.create_oval(tx-r, ty-r, tx+r, ty+r, outline=TARGET_COLOR, width=2, tags="markers")
    canvas.create_line(tx-r, ty, tx+r, ty, fill=TARGET_COLOR, tags="markers")
    canvas.create_line(tx, ty-r, tx, ty+r, fill=TARGET_COLOR, tags="markers")
    canvas.create_text(tx+13, ty-13, text="TARGET",
                       fill=TARGET_COLOR, font=("Courier", 9, "bold"), tags="markers")

    sx, sy = drone["starting_location"]
    canvas.create_rectangle(sx-5, sy-5, sx+5, sy+5, fill=START_COLOR, outline="", tags="markers")
    canvas.create_text(sx+11, sy-11, text="START",
                       fill=START_COLOR, font=("Courier", 9, "bold"), tags="markers")


# ─────────────────────────────────────────────
# DRONE SYMBOL
# ─────────────────────────────────────────────

def draw_drone_symbol(canvas, x, y, bearing_deg):
    ids = []
    r   = 6
    rad = math.radians(bearing_deg)
    nx  = x + math.cos(rad) * (r + 13)
    ny  = y + math.sin(rad) * (r + 13)
    ids.append(canvas.create_oval(x-r, y-r, x+r, y+r,
                                  fill=DRONE_COLOR, outline=""))
    ids.append(canvas.create_line(x, y, nx, ny, fill=DRONE_COLOR, width=2,
                                  arrow=tk.LAST, arrowshape=(8, 10, 4)))
    for sign in (+1, -1):
        wr = rad + sign * math.pi / 2
        ids.append(canvas.create_line(x, y,
                                      x + math.cos(wr) * 9,
                                      y + math.sin(wr) * 9,
                                      fill=DRONE_COLOR, width=1))
    return ids


# ─────────────────────────────────────────────
# FLAG SYMBOL
# ─────────────────────────────────────────────

def draw_flag(canvas, x, y, apparent_wind_angle):
    """Flag points downwind — opposite of apparent wind angle."""
    downwind_angle = (apparent_wind_angle + 180) % 360
    fx = x + math.cos(math.radians(downwind_angle)) * FLAG_LENGTH
    fy = y + math.sin(math.radians(downwind_angle)) * FLAG_LENGTH
    ids = []
    ids.append(canvas.create_line(x, y, fx, fy, fill=FLAG_COLOR, width=2))
    ids.append(canvas.create_oval(fx-3, fy-3, fx+3, fy+3,
                                  fill=FLAG_COLOR, outline=""))
    return ids


# ─────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────

def draw_legend(canvas, label=""):
    lx, ly = 10, CANVAS_H - 120
    canvas.create_rectangle(lx, ly, lx+200, ly+110,
                            fill="white", outline="#d3d1c7", tags="legend")
    canvas.create_line(lx+10, ly+15, lx+40, ly+15, fill=TRAIL_COLOR, width=2, tags="legend")
    canvas.create_text(lx+45, ly+15, text="flight path",
                       fill=TRAIL_COLOR, anchor="w", font=("Courier", 9), tags="legend")
    canvas.create_line(lx+10, ly+35, lx+25, ly+35, fill=WIND_COLOR,
                       arrow=tk.LAST, arrowshape=(4, 5, 2), tags="legend")
    canvas.create_text(lx+45, ly+35, text="wind",
                       fill=WIND_COLOR, anchor="w", font=("Courier", 9), tags="legend")
    canvas.create_rectangle(lx+13, ly+50, lx+21, ly+58, fill=START_COLOR, outline="", tags="legend")
    canvas.create_text(lx+45, ly+54, text="start",
                       fill=START_COLOR, anchor="w", font=("Courier", 9), tags="legend")
    canvas.create_oval(lx+10, ly+68, lx+24, ly+82, outline=TARGET_COLOR, width=2, tags="legend")
    canvas.create_text(lx+45, ly+75, text="target",
                       fill=TARGET_COLOR, anchor="w", font=("Courier", 9), tags="legend")
    canvas.create_line(lx+10, ly+95, lx+30, ly+95, fill=FLAG_COLOR, width=2, tags="legend")
    canvas.create_text(lx+45, ly+95, text="flag (downwind)",
                       fill=FLAG_COLOR, anchor="w", font=("Courier", 9), tags="legend")
    if label:
        canvas.create_text(lx+100, ly+5, text=label,
                           fill="#888", font=("Courier", 8), anchor="n", tags="legend")


# ─────────────────────────────────────────────
# CONTROLS PANEL
# ─────────────────────────────────────────────

def build_controls(root, canvas, state):

    panel = tk.Frame(root, bg="#e8e7e2", width=200)
    panel.pack(side=tk.RIGHT, fill=tk.Y)
    panel.pack_propagate(False)

    # ── PLACE ─────────────────────────────────
    tk.Label(panel, text="PLACE", bg="#e8e7e2", fg="#555",
             font=("Courier", 9, "bold")).pack(pady=(16, 4))

    place_status = tk.StringVar(value="")
    tk.Label(panel, textvariable=place_status, bg="#e8e7e2", fg="#888",
             font=("Courier", 7), wraplength=170).pack(padx=10)

    def start_place():
        state["place_clicks"] = []
        state["placing"]      = True
        place_status.set("click start on map")
        btn_place.config(bg="#f59e0b")

    btn_place = tk.Button(
        panel, text="place drone + target",
        command=start_place,
        bg="#374151", fg="#000", font=("Courier", 9, "bold"),
        relief=tk.FLAT, activebackground="#374151", activeforeground="#000",
        cursor="hand2"
    )
    btn_place.pack(fill=tk.X, padx=14, ipady=5, pady=4)

    def on_map_click(event):
        if not state.get("placing"):
            return
        clicks = state["place_clicks"]
        clicks.append((event.x, event.y))

        if len(clicks) == 1:
            place_status.set("click target on map")
            canvas.delete("place_preview")
            canvas.create_rectangle(
                event.x-5, event.y-5, event.x+5, event.y+5,
                fill=START_COLOR, outline="", tags="place_preview"
            )
        elif len(clicks) == 2:
            state["placing"]          = False
            state["start_pos"]        = list(clicks[0])
            state["target_pos"]       = list(clicks[1])
            place_status.set(f"start {clicks[0]}\ntarget {clicks[1]}")
            btn_place.config(bg="#374151")
            canvas.delete("place_preview")
            canvas.create_rectangle(
                clicks[0][0]-5, clicks[0][1]-5,
                clicks[0][0]+5, clicks[0][1]+5,
                fill=START_COLOR, outline="", tags="place_preview"
            )
            canvas.create_oval(
                clicks[1][0]-9, clicks[1][1]-9,
                clicks[1][0]+9, clicks[1][1]+9,
                outline=TARGET_COLOR, width=2, tags="place_preview"
            )

    canvas.bind("<Button-1>", on_map_click)

    # ── RUN ───────────────────────────────────
    tk.Label(panel, text="", bg="#e8e7e2").pack(pady=4)
    tk.Label(panel, text="SIMULATION", bg="#e8e7e2", fg="#555",
             font=("Courier", 9, "bold")).pack()

    speed_var = tk.IntVar(value=40)
    tk.Label(panel, text="playback speed", bg="#e8e7e2", fg="#555",
             font=("Courier", 7)).pack(pady=(8, 2))
    tk.Scale(panel, from_=1, to=50, variable=speed_var,
             orient=tk.HORIZONTAL, bg="#e8e7e2", highlightthickness=0,
             troughcolor="#ccc", showvalue=True, relief=tk.FLAT
             ).pack(fill=tk.X, padx=14)
    state["speed_var"] = speed_var

    tk.Button(
        panel, text="▶  RUN",
        command=lambda: state["run_callback"](),
        bg="#378add", fg="#000", font=("Courier", 12, "bold"),
        relief=tk.FLAT, activebackground="#378add", activeforeground="#000",
        cursor="hand2"
    ).pack(fill=tk.X, padx=14, ipady=8, pady=8)

    # ── CORRECTIONS ───────────────────────────
    tk.Label(panel, text="corrections", bg="#e8e7e2", fg="#555",
             font=("Courier", 9, "bold")).pack(pady=(4, 4))

    def toggle_corrections():
        simulation.CORRECTIONS = "ON" if simulation.CORRECTIONS == "OFF" else "OFF"
        btn_corrections.config(
            bg="#3a8a3a" if simulation.CORRECTIONS == "ON" else "#888",
            text=f"corrections: {simulation.CORRECTIONS}"
        )

    btn_corrections = tk.Button(
        panel,
        text=f"SENSORS: {simulation.CORRECTIONS}",
        command=toggle_corrections,
        bg="#3a8a3a" if simulation.CORRECTIONS == "ON" else "#888",
        fg="#000", font=("Courier", 9, "bold"),
        relief=tk.FLAT, activeforeground="#000",
        cursor="hand2"
    )
    btn_corrections.pack(fill=tk.X, padx=14, ipady=5, pady=4)

    # ── WIND ──────────────────────────────────
    tk.Label(panel, text="", bg="#e8e7e2").pack(pady=4)
    tk.Label(panel, text="WIND", bg="#e8e7e2", fg="#555",
             font=("Courier", 9, "bold")).pack()

    def wind_slider(label, attr, lo, hi, resolution=0.5):
        row = tk.Frame(panel, bg="#e8e7e2")
        row.pack(fill=tk.X, padx=14, pady=2)
        tk.Label(row, text=label, bg="#e8e7e2", fg="#555",
                 font=("Courier", 7), width=10, anchor="w").pack(side=tk.LEFT)
        var     = tk.DoubleVar(value=getattr(simulation, attr))
        val_lbl = tk.Label(row, bg="#e8e7e2", fg="#333",
                           font=("Courier", 7), width=5)
        val_lbl.pack(side=tk.RIGHT)

        def on_change(*_):
            setattr(simulation, attr, round(var.get(), 3))
            val_lbl.config(text=f"{var.get():.1f}")

        tk.Scale(row, from_=lo, to=hi, variable=var,
                 orient=tk.HORIZONTAL, bg="#e8e7e2",
                 highlightthickness=0, troughcolor="#ccc",
                 showvalue=False, relief=tk.FLAT,
                 resolution=resolution,
                 command=lambda _: on_change()
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        var.trace_add("write", on_change)
        on_change()

    wind_slider("direction", "WIND_MEAN_DIRECTION",      0,     360, 1)
    wind_slider("speed",     "WIND_MEAN_SPEED",          0,     15,  0.5)
    wind_slider("variation", "WIND_VARIATION",           0,     1,   0.05)
    wind_slider("dir vary",  "WIND_DIRECTION_VARIATION", 0,     180, 1)
    wind_slider("scale",     "WIND_SCALE",               0.001, 0.01, 0.001)

    # ── FLIGHT INFO ───────────────────────────
    tk.Label(panel, text="", bg="#e8e7e2").pack(pady=4)
    tk.Label(panel, text="FLIGHT INFO", bg="#e8e7e2", fg="#555",
             font=("Courier", 12, "bold")).pack()

    info_var = tk.StringVar(value="—")
    tk.Label(panel, textvariable=info_var, bg="#e8e7e2", fg="#333",
             font=("Courier", 12), justify=tk.LEFT, wraplength=170
             ).pack(padx=10, pady=4)
    state["info_var"] = info_var

    return panel


# ─────────────────────────────────────────────
# MAIN ANIMATE FUNCTION
# ─────────────────────────────────────────────

def animate_flightpath(drone, label=""):

    root   = tk.Tk()
    root.title("Dead Reckoning Drone Simulator")

    canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg=BG_COLOR)
    canvas.pack(side=tk.LEFT)

    state  = {
        "placing":      False,
        "place_clicks": [],
        "start_pos":    list(drone["starting_location"]),
        "target_pos":   list(drone["target_location"]),
        "anim_id":      None,
    }

    build_controls(root, canvas, state)

    anim_ids = []

    def redraw_static(d):
        canvas.delete("wind_field")
        canvas.delete("markers")
        canvas.delete("legend")
        canvas.delete("trail")
        draw_wind_field(canvas)
        draw_markers(canvas, d)
        draw_legend(canvas, label=label)

    def run_simulation():
        # clear previous flight
        canvas.delete("trail")
        canvas.delete("place_preview")
        for item in anim_ids:
            canvas.delete(item)
        anim_ids.clear()

        # build and run drone
        d = create_drone()
        program_drone(d, state["start_pos"], state["target_pos"])
        compute_flight_plan(d)
        fly(d)
        error_before = get_distance_to_target(d)
        final_approach(d)
        error_after = get_distance_to_target(d)

        redraw_static(d)

        flightpath     = d["location_log"]
        wind_angle_log = d["apparent_wind_angle_log"]
        bearingpath    = d["bearingpath"]
        final_error    = get_distance_to_target(d)

        step = [0]

        def draw_step():
            i = step[0]
            if i >= len(flightpath) - 1:
                total_dist = initial_distance_to_target(d)
                state["info_var"].set(
                    f"Total Distance:\n{total_dist:.0f} m\n\n"
                    f"Lnding error:\n{final_error:.1f} m\n\n"
                    f"Error before AI-assist: {error_before:.1f} m\n\n"
                    f"Error after AI-assist:  {error_after:.1f} m\n\n"
                    f"improvement: {error_before - error_after:.1f} m\n\n"
                    f"steps: {len(d['location_log'])}"
                )
                return

            x, y           = flightpath[i]
            apparent_angle = wind_angle_log[i]
            bearing        = bearingpath[i]

            if i % 10 == 0:
                canvas.create_oval(x-2, y-2, x+2, y+2,
                                   fill=TRAIL_COLOR, outline="", tags="trail")

            for item in anim_ids:
                canvas.delete(item)
            anim_ids.clear()

            anim_ids.extend(draw_drone_symbol(canvas, x, y, bearing))
            anim_ids.extend(draw_flag(canvas, x, y, apparent_angle))

            state["info_var"].set(
                f"step: {i}\n"
                f"bearing: {bearing:.1f}°\n"
                f"apparent wind: {apparent_angle:.1f}°\n"
                f"pos: ({x:.0f}, {y:.0f})"
            )

            step[0] += state["speed_var"].get()
            root.after(16, draw_step)

        draw_step()

    state["run_callback"] = run_simulation

    # initial draw
    redraw_static(drone)

    root.mainloop()