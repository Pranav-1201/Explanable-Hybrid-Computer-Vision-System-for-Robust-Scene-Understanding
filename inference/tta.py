import torch
import torchvision.transforms as T
from contextlib import nullcontext

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TTA_TRANSFORMS = [
    # 1. Standard center crop
    T.Compose([
        T.Resize((232, 232)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # 2. Horizontal flip + center crop
    T.Compose([
        T.Resize((232, 232)),
        T.RandomHorizontalFlip(p=1.0),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # 3. Five-crop average
    T.Compose([
        T.Resize((256, 256)),
        T.FiveCrop(224),
        T.Lambda(lambda crops: torch.stack([
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(T.ToTensor()(c))
            for c in crops
        ])),
    ]),
]

def tta_predict(model, pil_image, device, scaler=None):
    """
    Run TTA inference. Returns averaged softmax probability vector (num_classes,).
    Falls back to single-crop if model is in training mode.
    """
    model.eval()
    preds = []
    use_amp = torch.cuda.is_available()
    amp_ctx = torch.amp.autocast('cuda') if use_amp else nullcontext()

    with torch.no_grad(), amp_ctx:
        for tfm in TTA_TRANSFORMS:
            t = tfm(pil_image)
            if t.dim() == 4:          # FiveCrop: (5, C, H, W)
                t = t.to(device)
                logits = model(t).mean(0)          # average over 5 crops
            else:
                logits = model(t.unsqueeze(0).to(device))[0]
            if scaler is not None:
                probs_i = scaler.calibrate_probs(logits.unsqueeze(0))[0]
            else:
                probs_i = torch.softmax(logits, dim=0)
            preds.append(probs_i)

    return torch.stack(preds).mean(0).cpu()       # (num_classes,)


def single_predict(model, pil_image, device, scaler=None):
    """Standard single-crop inference for speed comparison."""
    tfm = TTA_TRANSFORMS[0]
    model.eval()
    with torch.no_grad():
        t = tfm(pil_image).unsqueeze(0).to(device)
        logits = model(t)[0]
    if scaler is not None:
        return scaler.calibrate_probs(logits.unsqueeze(0))[0].cpu()
    return torch.softmax(logits, dim=0).cpu()
