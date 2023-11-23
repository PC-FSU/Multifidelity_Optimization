import torch

NUM_RESTARTS = 10 
RAW_SAMPLES = 512

tkwargs = {
    "dtype":  torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}