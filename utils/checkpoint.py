import torch
import os


def _match_prefix(state_dict, model):
    """Reconcile the 'model.' key-prefix difference between wrapped models
    (e.g. CNNBaseline, whose keys look like 'model.layer4...') and unwrapped
    torchvision models ('layer4...'), in whichever direction is needed.

    Returns a state_dict whose keys align with `model`. If no alignment is
    possible the original dict is returned unchanged, so that a subsequent
    strict load_state_dict raises a precise, loud error (audit N11) rather
    than silently loading nothing.
    """
    model_keys = set(model.state_dict().keys())
    if set(state_dict.keys()) & model_keys:
        return state_dict  # already aligned

    stripped = {k[len('model.'):]: v for k, v in state_dict.items() if k.startswith('model.')}
    if stripped and (set(stripped) & model_keys):
        return stripped

    added = {('model.' + k): v for k, v in state_dict.items()}
    if set(added) & model_keys:
        return added

    return state_dict


def load_checkpoint(path: str, model: torch.nn.Module, device=None,
                    load_ema: bool = False, strict: bool = True) -> torch.nn.Module:
    """Load weights into `model` from any checkpoint format this project
    produces, and return it in eval() mode.

    Recognised container formats (checked in order):
      - {'model_state': ..., 'ema_state': ...}   (training/train_phase2.py)
      - {'state_dict': ...}                       (generic)
      - raw state_dict                            (train_baseline / transfer / hybrid)

    Args:
        path:     Path to the checkpoint file.
        model:    Instantiated model whose architecture matches the checkpoint.
        device:   Target device. Defaults to CUDA if available, else CPU.
        load_ema: When the checkpoint carries EMA weights ('ema_state'),
                  load those instead of the raw weights. Falls back to
                  'model_state' with a warning if no EMA weights are present
                  (audit N5 — this parameter previously did not exist).
        strict:   Passed to load_state_dict. Default True so architecture
                  mismatches fail loudly instead of loading a partial/garbage
                  state dict (audit N11). Pass strict=False only when a partial
                  load is intended.

    Returns:
        model with loaded weights (in-place), on `device`, in eval() mode.

    Raises:
        FileNotFoundError: if path does not exist.
        RuntimeError:      if the state dict is incompatible under `strict`.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # weights_only=True: checkpoints are tensors + primitives only; refuse to
    # unpickle arbitrary objects (safe against tampered checkpoint files).
    ckpt = torch.load(path, map_location=device, weights_only=True)

    # 1) Extract the state_dict from whatever container this is.
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        ema = ckpt.get('ema_state')
        if load_ema:
            if ema is None:
                print(f"[WARN] load_ema=True but '{path}' has no ema_state; using model_state.")
                state_dict = ckpt['model_state']
            else:
                state_dict = ema
        else:
            state_dict = ckpt['model_state']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        if load_ema:
            print(f"[WARN] load_ema=True but '{path}' has no ema_state; using state_dict.")
    else:
        state_dict = ckpt
        if load_ema:
            print(f"[WARN] load_ema=True but '{path}' is a bare state_dict; no EMA weights.")

    # 2) Reconcile 'model.' prefix in whichever direction is needed.
    state_dict = _match_prefix(state_dict, model)

    # 3) Load.
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    return model
