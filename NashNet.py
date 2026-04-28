import torch 
from torch import nn
import os
_DIR = os.path.dirname(os.path.abspath(__file__))


class NashPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 3)
        )

    def forward(self, x):
        return self.network(x)


SPEED_MAP = {
    0: (-0.1,  0.2),   # case 0: lane=slow, merger=slow
    1: (-0.1,  0.15),  # case 1: lane=slow, merger=fast
    2: (-0.15, 0.2),   # case 2: lane=fast, merger=slow
    3: (-0.15, 0.15),  # case 3: both fast
}

_model = None
def get_speeds(real_dist_m):
    global _model
    if _model is None:
        _model = NashPredictor()
        _model.load_state_dict(torch.load(os.path.join(_DIR, 'nash-weights.pt'), weights_only=True))
        _model.eval()

    # Map real distance [0.6, 1.2] → simulation range [1.0, 2.0]
    sim_dist = 1.0 + (real_dist_m - 0.6) / (1.2 - 0.6)
    sim_dist = max(1.0, min(2.0, sim_dist))  # clamp

    x = torch.tensor([[sim_dist]], dtype=torch.float32)
    with torch.no_grad():
        out = _model(x)
    best_case = torch.argmin(out).item()  # smallest dx = safest merge gap
    return SPEED_MAP[best_case]