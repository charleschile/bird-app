# config.py

import torch

class CFG:
    sample_rate = 32000
    n_mels = 256
    fmin = 80
    fmax = 15000
    duration = 10.0
    model_name = "tf_efficientnet_b2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
