"""Optional random-state controls for paired Stage-2 experiments."""

from contextlib import contextmanager
import random

import numpy as np
import torch


def reset_training_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def preserve_rng_state(enabled=True):
    """Restore RNG streams after graph extraction, including on failure.

    Only already-initialized CUDA generators are read; CPU diagnostics do not
    create a CUDA context. DataLoader worker seeds derive from the restored
    PyTorch CPU generator on the next training iterator.
    """
    if not enabled:
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
