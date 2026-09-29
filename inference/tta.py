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


def _probs_from_logits(logits, scaler):
    if scaler is not None:
        return scaler.calibrate_probs(logits)
    return torch.softmax(logits, dim=-1)


def _amp():
    return torch.amp.autocast('cuda') if torch.cuda.is_available() else nullcontext()


def tta_predict_batch(model, pil_images, device, scaler=None, max_images=8):
    """tta_predict for many images with the views of ALL images stacked into
    one forward pass (M10). Same maths as calling tta_predict per image -- each
    image contributes 1 + 1 + 5 tensors, the five crops are averaged as logits,
    every view is calibrated on its own and the three views are averaged -- but
    the GPU/CPU sees one large batch instead of 3 small ones per image.

    Returns (N, num_classes), row i belonging to pil_images[i]. `max_images`
    bounds how many images share one forward pass (7 tensors each), so a large
    request cannot allocate an unbounded activation tensor.
    """
    if not pil_images:
        raise ValueError("tta_predict_batch needs at least one image")
    model.eval()
    out = []
    for start in range(0, len(pil_images), max_images):
        chunk = pil_images[start:start + max_images]
        views = []
        for im in chunk:
            v1, v2, v3 = (t(im) for t in TTA_TRANSFORMS)      # v3 is (5, C, H, W)
            views.extend([v1.unsqueeze(0), v2.unsqueeze(0), v3])
        x = torch.cat(views).to(device)                        # (7 * len(chunk), C, H, W)
        with torch.no_grad(), _amp():
            logits = model(x).float().reshape(len(chunk), 7, -1)
        per_view = torch.stack([logits[:, 0], logits[:, 1], logits[:, 2:].mean(1)], dim=1)
        out.append(_probs_from_logits(per_view, scaler).mean(1).cpu())
    return torch.cat(out)


def single_predict_batch(model, pil_images, device, scaler=None, max_images=32):
    """single_predict for many images in one forward pass. (N, num_classes)."""
    if not pil_images:
        raise ValueError("single_predict_batch needs at least one image")
    model.eval()
    out = []
    for start in range(0, len(pil_images), max_images):
        chunk = pil_images[start:start + max_images]
        x = torch.stack([TTA_TRANSFORMS[0](im) for im in chunk]).to(device)
        with torch.no_grad():
            logits = model(x).float()
        out.append(_probs_from_logits(logits, scaler).cpu())
    return torch.cat(out)
