# rl_model.py

import torch
import torch.nn as nn

INPUT_SIZE  = 3
HIDDEN_SIZE = 64
OUTPUT_SIZE = 2

class FinalApproachRNN(nn.Module):

    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super().__init__()
        self.gru    = nn.GRU(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        correction = self.output(out[:, -1, :])
        return correction


def build_input_tensor(drone):
    wind_angles = drone["apparent_wind_angle_log"]
    wind_speeds = drone["apparent_wind_speed_log"]
    bearings    = drone["bearingpath"]

    seq = [
        [
            wind_angles[i] / 360.0,
            wind_speeds[i] / 10.0,
            bearings[i]    / 360.0,
        ]
        for i in range(len(wind_angles))
    ]
    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0)


_model = None

def predict(drone):
    if _model is None:
        print("rl_model: no weights loaded, returning zero correction")
        return 0.0, 0.0
    x = build_input_tensor(drone)
    with torch.no_grad():
        out = _model(x)
    delta_bearing = out[0, 0].item() * 180.0
    extra_time    = max(0.0, out[0, 1].item() * 100.0)
    return delta_bearing, extra_time

def save_model(path="rl_weights.pt"):
    if _model is None:
        print("no model to save")
        return
    torch.save(_model.state_dict(), path)
    print(f"saved → {path}")

def load_model(path="rl_weights.pt"):
    global _model
    _model = FinalApproachRNN()
    _model.load_state_dict(torch.load(path, weights_only=True))
    _model.eval()
    print(f"loaded ← {path}")