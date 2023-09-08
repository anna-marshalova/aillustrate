import gc
import torch

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()