import torch
import os

def load_checkpoint(path: str, model: torch.nn.Module, device=None) -> torch.nn.Module:
    """
    Load a model checkpoint with automatic 'model.' prefix key resolution.
    
    Handles the common key mismatch where checkpoints saved from a wrapped model
    have keys prefixed with 'model.' that don't match the target model's state dict.

    Args:
        path:   Path to the .pth checkpoint file.
        model:  Instantiated model (architecture must already match checkpoint).
        device: Target device. Defaults to CUDA if available, else CPU.

    Returns:
        model with loaded weights (in-place), set to eval() mode.

    Raises:
        FileNotFoundError: if path does not exist.
        RuntimeError: if no keys match after prefix stripping (fully incompatible checkpoint).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path, map_location=device)

    # Support both raw state_dicts and checkpoint dicts with a 'state_dict' key
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Resolve 'model.' prefix mismatch
    model_keys = set(model.state_dict().keys())
    ckpt_keys  = set(state_dict.keys())

    if not model_keys.intersection(ckpt_keys):
        # Try stripping 'model.' prefix
        stripped = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
        # Try adding 'model.' prefix
        added = {('model.' + k) if not k.startswith('model.') else k: v for k, v in state_dict.items()}
        
        if model_keys.intersection(stripped.keys()):
            state_dict = stripped
        elif model_keys.intersection(added.keys()):
            state_dict = added
        else:
            raise RuntimeError(
                f"Checkpoint at '{path}' has no matching keys with this model. "
                f"Sample checkpoint keys: {list(state_dict.keys())[:5]}"
            )

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model
