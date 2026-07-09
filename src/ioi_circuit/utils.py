import torch
import gc
import psutil
import os
import time
import random
import numpy as np


def safe_div(a, b, eps=1e-8):
    return a / (b + eps) if abs(b) > eps else 0.0


def jaccard(s1, s2):
    a, b = set(s1), set(s2)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def name_tokens(tokenizer, name):
    """Token ids for a name, with a leading space like the model sees it.
    Returns all subtokens so multi-token names (e.g. a name split across
    subwords) are scored correctly instead of only the first subtoken.
    """
    ids = tokenizer.encode(" " + name, add_special_tokens=False)
    if not ids:
        raise ValueError(f"name {name!r} produced no tokens")
    return ids


def logit_diff(logits, cid, wid):
    """Mean logit difference between the correct and wrong name tokens.
    cid/wid are lists of token ids (multi-token names are averaged).
    """
    last = logits[:, -1, :]
    ct = torch.tensor(cid, device=last.device)
    wt = torch.tensor(wid, device=last.device)
    c = last.gather(1, ct.unsqueeze(1)).mean(dim=1)
    w = last.gather(1, wt.unsqueeze(1)).mean(dim=1)
    return (c - w).mean().item()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name, device, dtype_str):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtypes = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtypes.get(dtype_str, torch.float16)

    if device == "auto":
        # auto device selection, MPS on my machine
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"loading {model_name} on {device} ({dtype_str})")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded in {time.time()-t0:.1f}s, {n_params/1e6:.0f}M params")
    return model, tokenizer, device, dtype


def mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"unsupported model architecture: {type(model).__name__}")


def n_layers(model):
    return len(get_layers(model))
